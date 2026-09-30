import numpy as np
import joblib
import os
import json
import re
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

# Load Production Artifacts
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "Execution"
MODEL_PATH = ARTIFACT_DIR / "risk_model.pkl"
ENCODER_PATH = ARTIFACT_DIR / "label_encoders.pkl"
NLP_ANCHORS_PATH = ARTIFACT_DIR / "nlp_anchors.pt"
CALIBRATION_PATH = ARTIFACT_DIR / "calibration_profiles.json"

class ThreatIntelligencePredictor:
    """
    Supplychainer Quantile ML Decision Brain.
    V3: Statistically Defensible Calibration & Geographic Hub Intelligence.
    """
    def __init__(self, lazy_load=False):
        self.is_trained = False
        self.model = None
        self.encoders = None
        self.profiles = {}
        
        self.hub_map = {
            "Seattle": "Seattle Port", "Portland": "Portland Terminal", "San Francisco": "San Francisco Port",
            "Los Angeles": "Los Angeles Port", "Salt Lake City": "Salt Lake City Hub", "Denver": "Denver Terminal",
            "Phoenix": "Phoenix Logistics", "Dallas": "Dallas Corridor", "Houston": "Houston Port",
            "Chicago": "Chicago Rail Hub", "St. Louis": "St. Louis Hub", "Atlanta": "Atlanta Air Hub",
            "Miami": "Miami Port", "New York": "New York Port", "Boston": "Boston Terminal",
            "Shanghai": "Shanghai Port", "Rotterdam": "Rotterdam Port",
            "Singapore": "Singapore Port", "Dubai": "Dubai Logistics Hub",
            "Mumbai": "Mumbai Port", "Kochi": "Kochi Port", "Delhi": "Delhi Air Cargo", "Chennai": "Chennai Port"
        }
        
        if not lazy_load:
            self.warmup()

    def warmup(self):
        if self.is_trained: return
        print("[PREDICTOR] Starting warmup...")
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
            print(f"CRITICAL: Production models missing. Running in deterministic fallback mode.")
            return
            
        # 1. Load ML Core
        self.model = joblib.load(MODEL_PATH)
        self.encoders = joblib.load(ENCODER_PATH)
        self.is_trained = True
        
        # 2. Load Statistically Defensible Calibration Profiles
        if os.path.exists(CALIBRATION_PATH):
            with open(CALIBRATION_PATH, 'r') as f:
                self.profiles = json.load(f)
            print(f"Calibration Layer: Loaded {len(self.profiles)} mode profiles from historical p5/p95 analysis.")
        else:
            print("WARNING: Calibration profiles missing. Using defensive fallbacks.")
            self.profiles = {}

        print(f"Supplychainer V3 Brain Loaded: Production-Ready.")

    def _resolve_feature(self, value: str, key: str):
        classes = list(self.encoders[key].classes_)
        resolved = self.hub_map.get(value, value) if key.endswith("_Node") else value
        if resolved in classes:
            return resolved, False
        # Explicit generic hub categories instead of silently using the first city.
        generic = {"Origin_Node": "Regional Hub", "Destination_Node": "Local Terminal"}
        if key in generic and generic[key] in classes:
            return generic[key], True
        raise ValueError(f"Unsupported {key}: {value}")

    def _encode_feature(self, value: str, key: str) -> int:
        resolved, _ = self._resolve_feature(value, key)
        return int(self.encoders[key].transform([resolved])[0])

    @staticmethod
    def _operational_prior(transport_mode, reason):
        delay = {"road": 2.5, "sea": 48.0, "air": 12.0, "rail": 18.0}.get(transport_mode.lower(), 12.0)
        return {"raw_model_prediction": None, "calibrated_delay": delay,
                "baseline_systemic_friction": delay, "final_delay_presented": delay,
                "calibration_reason": reason, "p_quantile": None,
                "prediction_source": "OPERATIONAL_PRIOR", "is_defensible": False,
                "unsupported_features": []}

    def predict_worst_case_delay(self, origin: str, destination: str, transport_mode: str,
                                 leg_type: str = "Global_Freight", condition_flag: str = "Clear",
                                 nlp_score: float = 0.0) -> Dict[str, Any]:
        return self.predict_worst_case_delays([dict(origin=origin, destination=destination,
            transport_mode=transport_mode, leg_type=leg_type, condition_flag=condition_flag,
            nlp_score=nlp_score)])[0]

    def predict_worst_case_delays(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch p85 inference so a graph request does not call sklearn thousands of times."""
        results = [None] * len(requests)
        if not self.is_trained:
            return [self._operational_prior(r["transport_mode"], "Model unavailable; operational prior")
                    for r in requests]
        rows, indices, unsupported = [], [], []
        for i, request in enumerate(requests):
            try:
                features = {"Leg_Type": request.get("leg_type", "Global_Freight"),
                            "Origin_Node": request["origin"], "Destination_Node": request["destination"],
                            "Transport_Mode": request["transport_mode"].lower(),
                            "Condition_Flag": request.get("condition_flag", "Clear")}
                row, unknown = {}, []
                for key, value in features.items():
                    resolved, substituted = self._resolve_feature(value, key)
                    row[key] = int(self.encoders[key].transform([resolved])[0])
                    if substituted:
                        unknown.append(key)
                row["NLP_Severity_Score"] = float(np.clip(request.get("nlp_score", 0.0), 0, 1))
                rows.append(row); indices.append(i); unsupported.append(unknown)
            except Exception as exc:
                results[i] = self._operational_prior(request["transport_mode"], f"Inference fallback: {exc}")
        if rows:
            try:
                raw_predictions = self.model.predict(pd.DataFrame(rows))
                for i, raw, unknown in zip(indices, raw_predictions, unsupported):
                    if not np.isfinite(raw):
                        raise ValueError("Non-finite model prediction")
                    profile = self.profiles.get(requests[i]["transport_mode"].lower(), {"floor": 0.0, "cap": 240.0})
                    floor, cap = profile["floor"], profile["cap"]
                    calibrated = min(max(0.0, float(raw)), cap)
                    final = max(calibrated, floor)
                    reason = "Quantile Disruption Prediction (p85 Risk)"
                    if calibrated < floor:
                        reason = f"Baseline Operational Friction (floor: {floor}h)"
                    elif raw > cap:
                        reason = f"Operational Cap Applied ({cap}h)"
                    if unknown:
                        reason += "; generic hub encoding used"
                    results[i] = {"raw_model_prediction": round(float(raw), 2),
                        "calibrated_delay": round(calibrated, 2), "baseline_systemic_friction": floor,
                        "final_delay_presented": round(final, 2), "calibration_reason": reason,
                        "p_quantile": 0.85, "prediction_source": "ML_P85",
                        "unsupported_features": unknown, "is_defensible": not bool(unknown)}
            except Exception as exc:
                for i in indices:
                    results[i] = self._operational_prior(requests[i]["transport_mode"], f"Inference fallback: {exc}")
        return results

class ContrastiveNLPEngine:
    """Stage 2: PRODUCTION Contrastive NLP Brain."""
    def __init__(self, lazy_load=False):
        self._ready = False
        self.noise_floor = 0.04
        self.calibration_multiplier = 0.35
        if not lazy_load:
            self.warmup()

    def warmup(self):
        if self._ready: return
        print("[NLP ENGINE] Starting warmup...")
        try:
            import torch
            from sentence_transformers import SentenceTransformer, util
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.util = util
            if os.path.exists(NLP_ANCHORS_PATH):
                anchors = torch.load(NLP_ANCHORS_PATH)
                self.disaster_matrix = anchors["disaster_matrix"]
                self.safe_matrix = anchors["safe_matrix"]
                self._ready = True
                print(f"NLP Brain: Loaded Historical Anchor Matrix.")
            else:
                self._ready = False
        except Exception as e:
            print(f"[NLP ENGINE] Warmup failed: {e}")
            self._ready = False

    def get_semantic_score(self, news_text: str) -> float:
        # t_nlp_start = time.perf_counter()
        if not self._ready: return 0.0
        if not news_text or len(news_text.strip()) < 5: return 0.0
        chunks = [news_text[i:i+256] for i in range(0, len(news_text), 256)]
        chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
        d_scores = self.util.cos_sim(chunk_embeddings, self.disaster_matrix)
        s_scores = self.util.cos_sim(chunk_embeddings, self.safe_matrix)
        # Compare disaster and safe anchors within each chunk, then retain the strongest threat.
        margins = d_scores.cpu().numpy().max(axis=1) - s_scores.cpu().numpy().max(axis=1)
        margin = float(np.max(margins))
        if margin <= self.noise_floor: return 0.0
        return float(np.clip((margin - self.noise_floor) * self.calibration_multiplier, 0.0, 1.0))

class CARFFilter:
    """Stage 3: TRUE CARF (Context-Aware Relevance Filter)."""
    def __init__(self):
        self.relevance_map = {"air": ["airport", "flight", "airspace", "aviation", "sky", "terminal"],
                              "sea": ["port", "vessel", "ship", "canal", "ocean", "maritime", "dock"],
                              "rail": ["rail", "track", "locomotive", "station"],
                              "road": ["highway", "truck", "traffic", "bridge", "road", "delivery"]}

    def apply_filter(self, semantic_score: float, news_context: str, transport_mode: str) -> float:
        mode = transport_mode.lower()
        if semantic_score <= 0 or mode not in self.relevance_map:
            return 0.0
        words = set(re.findall(r"[a-z]+", news_context.lower()))
        # Match punctuation and simple plurals, but never substring-match airport as port.
        words |= {w[:-1] for w in words if w.endswith("s")}
        if not words.intersection(self.relevance_map[mode]):
            return 0.0
        return float(np.clip(semantic_score, 0.0, 1.0))

    def max_pool_threats(self, scores: List[float]) -> float:
        return float(np.max(scores)) if scores else 0.0
