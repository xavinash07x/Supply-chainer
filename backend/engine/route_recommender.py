import networkx as nx
import math
from .multimodal_network import MODE_PROFILES, create_multimodal_network
from .threat_intelligence import ContrastiveNLPEngine, CARFFilter
from .news_ingestion import DynamicNewsIngestor
from .node_resolver import NodeResolver

class RouteRecommender:
    """
    Supplychainer Unified Multimodal Optimization Engine.
    V8: Virtual-Node Forensic Edition.
    """

    def __init__(self, network, predictor, simulator, scenario_mgr, demo_mode=False):
        self.network = network # Legacy
        self.predictor = predictor
        self.simulator = simulator
        self.scenario_mgr = scenario_mgr
        self.demo_mode = demo_mode
        self.is_warmed_up = False
        self.warmup_failed = False
        
        self.nlp = ContrastiveNLPEngine(lazy_load=True)
        self.carf = CARFFilter()
        self.news_ingestor = DynamicNewsIngestor()
        self.resolver = NodeResolver()
        
        print(f"[STARTUP] Initializing Split-Node Global Topology...")
        self.unified_graph = create_multimodal_network()
        
        if self.demo_mode:
            self.is_warmed_up = True
            
        print(f"[STARTUP] Unified Engine Ready.")

    def run_background_warmup(self):
        if self.is_warmed_up: return
        print("[WARMUP] Calibrating global threat floor...")
        try:
            self.predictor.warmup()
            self.nlp.warmup()
            
            # Only four fallback texts exist; score each once, not once per edge.
            baseline = {}
            for mode, news in self.news_ingestor.fallback_news.items():
                baseline[mode] = (news, self.carf.apply_filter(self.nlp.get_semantic_score(news), news, mode))
            for u, v, d in self.unified_graph.edges(data=True):
                mode = d.get("transport_mode", "road")
                if mode == "transfer": continue
                news, threat = baseline.get(mode, ("Normal conditions.", 0.0))
                self.unified_graph[u][v]["base_threat"] = threat
                self.unified_graph[u][v]["base_news"] = news
                
            self.is_warmed_up = True
            print("[WARMUP] Unified Calibration Complete.")
        except Exception as e:
            print(f"[WARMUP] Error during warmup: {e}")
            self.warmup_failed = True

    def _prepare_intelligence(self, graph, disruptions):
        """Resolve each edge once; the solver and response share the same delay numbers."""
        requests, edges = [], []
        for u, v, edge in graph.edges(data=True):
            mode = edge["transport_mode"]
            baseline_threat = max(0.0, min(1.0, edge.get("base_threat", edge.get("risk", 0.0))))
            disruption = disruptions.get(graph.nodes[v].get("physical_id"), {})
            if mode == "transfer" or disruption.get("mode") != mode:
                disruption = {}
            threat = max(baseline_threat, disruption.get("threat", 0.0))
            edge["intelligence"] = {
                "baseline_threat": baseline_threat, "threat": threat,
                "scenario_delay": disruption.get("delay", 0.0),
                "reason": disruption.get("reason", edge.get("base_news", "Standard conditions")),
                "intel_source": "SCENARIO" if disruption else "FALLBACK",
                "prediction": None, "model_delay": 0.0, "scenario_increment": 0.0,
            }
            if mode == "transfer":
                continue
            def model_location(node):
                data = graph.nodes[node]
                return data.get("parent_city") or data.get("display_name", node)
            requests.append(dict(origin=model_location(u), destination=model_location(v),
                transport_mode=mode, leg_type="Global_Freight" if mode in ("air", "sea") else "Last_Mile",
                condition_flag="Clear", nlp_score=threat))
            edges.append(edge)
        predictions = self.predictor.predict_worst_case_delays(requests)
        for edge, prediction in zip(edges, predictions):
            intel = edge["intelligence"]
            delay = float(prediction["final_delay_presented"])
            if not math.isfinite(delay) or delay < 0:
                raise ValueError("Delay predictor returned an invalid buffer")
            intel["prediction"] = prediction
            intel["model_delay"] = delay
            # The model already sees scenario severity. Treat a scripted delay as a
            # minimum buffer; adding the full scenario would count the same risk twice.
            intel["scenario_increment"] = max(0.0, intel["scenario_delay"] - delay)

    def recommend(self, source: str, destination: str, transport_preference: str = "any",
                  routing_policy: str = "STRICT", cargo_type: str = "general",
                  priority: str = "normal", scenario: str = None,
                  overrides: dict = None) -> dict:
        if transport_preference not in {"any", "air", "sea", "rail", "road"}:
            return {"error": "Unknown transport preference"}
        if routing_policy not in {"STRICT", "PREFERRED"}:
            return {"error": "Unknown routing policy"}
        if scenario and scenario not in self.scenario_mgr.SCENARIOS:
            return {"error": "Unknown disruption scenario"}
        overrides = overrides or {}
        avoid_hubs = overrides.get("avoid_chokepoints", [])
        cost_ceiling = overrides.get("cost_ceiling", 999999)
        max_delay = overrides.get("max_delay", 9999)
        res_s = self.resolver.resolve_node_to_entry_point(source)
        res_d = self.resolver.resolve_node_to_entry_point(destination)
        if "error" in res_s: return {"error": res_s["error"]}
        if "error" in res_d: return {"error": res_d["error"]}
        s_vnode, d_vnode = res_s["id"], res_d["id"]

        # Request-local scenarios cannot be overwritten by a concurrent request.
        active_scenario = self.scenario_mgr.SCENARIOS.get(scenario)
        disruptions = self.scenario_mgr.get_disruptions(scenario)
        graph = self.unified_graph.copy()
        graph.remove_nodes_from([n for n, d in graph.nodes(data=True)
                                 if d.get("physical_id") in avoid_hubs])
        remove_edges = []
        for u, v, edge in graph.edges(data=True):
            mode = edge["transport_mode"]
            if cargo_type in MODE_PROFILES.get(mode, {}).get("cargo_restrictions", []):
                remove_edges.append((u, v))
            elif transport_preference != "any" and routing_policy == "STRICT":
                if mode not in {transport_preference, "transfer", "road"}:
                    remove_edges.append((u, v))
        graph.remove_edges_from(remove_edges)
        if s_vnode not in graph or d_vnode not in graph:
            return {"error": "An endpoint is excluded by the hub avoidance constraints"}
        self._prepare_intelligence(graph, disruptions)

        candidates = []
        for persona in ["FASTEST", "SAFEST", "BALANCED"]:
            def weight_func(u, v, edge):
                intel = edge["intelligence"]
                eta = edge["baseline_time"] + intel["model_delay"] + intel["scenario_increment"]
                threat = intel["threat"]
                if persona == "FASTEST":
                    weight = eta
                elif persona == "SAFEST":
                    weight = eta * (1.0 + threat * 12.0)
                else:
                    weight = eta * 0.3 + edge.get("cost", 0) / 150.0 * 0.5 + threat * 40.0 * 0.2
                if routing_policy == "PREFERRED" and transport_preference != "any":
                    if edge["transport_mode"] not in {transport_preference, "transfer", "road"}:
                        weight *= 1.25
                return weight
            try:
                path = nx.dijkstra_path(graph, s_vnode, d_vnode, weight=weight_func)
            except nx.NetworkXNoPath:
                continue
            legs, transfer_count = [], 0
            trace = {"eta": {"transit": 0.0, "transfer": 0.0, "model": 0.0, "scenario": 0.0},
                     "cost": {"transit": 0.0, "transfer": 0.0, "scenario": 0.0},
                     "risk": {"baseline": 0.0, "scenario": 0.0}}
            for u, v in zip(path, path[1:]):
                edge, v_data = graph[u][v], graph.nodes[v]
                intel = edge["intelligence"]
                component = "transfer" if edge["type"] == "transfer" else "transit"
                transfer_count += int(component == "transfer")
                trace["eta"][component] += edge["baseline_time"]
                trace["eta"]["model"] += intel["model_delay"]
                trace["eta"]["scenario"] += intel["scenario_increment"]
                trace["cost"][component] += edge.get("cost", 0.0)
                trace["risk"]["baseline"] = max(trace["risk"]["baseline"], intel["baseline_threat"])
                if intel["intel_source"] == "SCENARIO":
                    trace["risk"]["scenario"] = max(trace["risk"]["scenario"], intel["threat"])
                legs.append({
                    "from": graph.nodes[u].get("physical_id", u),
                    "to": v_data.get("physical_id", v),
                    "to_name": v_data.get("display_name", v),
                    "mode": edge["transport_mode"].upper(), "type": edge["type"],
                    "eta": round(edge["baseline_time"] + intel["model_delay"] + intel["scenario_increment"], 1),
                    "baseline_eta": round(edge["baseline_time"], 1),
                    "model_delay": intel["model_delay"], "scenario_delay": intel["scenario_delay"],
                    "scenario_increment": round(intel["scenario_increment"], 2),
                    "cost": round(edge.get("cost", 0.0), 2), "threat": round(intel["threat"], 2),
                    "reason": intel["reason"], "intel_source": intel["intel_source"],
                    "delay_prediction": intel["prediction"]})
            total_time, total_cost = sum(trace["eta"].values()), sum(trace["cost"].values())
            if total_cost > cost_ceiling or total_time > max_delay * 24:
                continue
            max_threat = max((l["threat"] for l in legs), default=0.0)
            candidates.append({"persona": persona, "primary_mode": "MULTIMODAL", "legs": legs,
                "adjusted_eta": round(total_time, 1), "total_cost": round(total_cost, 2),
                "threat_level": max_threat, "audit_trace": trace,
                "explanation": self._generate_forensic_explanation(persona, trace, max_threat, transfer_count),
                "override_applied": bool(overrides),
                "delay_method": "Sum of per-leg calibrated buffers; not a route-level p85 guarantee"})
        if not candidates:
            return {"error": "No valid multimodal route under current strategic constraints"}
        final, seen = [], set()
        for candidate in sorted(candidates, key=lambda x: x["adjusted_eta"]):
            signature = tuple((leg["from"], leg["to"], leg["mode"]) for leg in candidate["legs"])
            if signature not in seen:
                final.append(candidate); seen.add(signature)
        return {"origin": source, "destination": destination,
                "active_scenario": active_scenario["name"] if active_scenario else None,
                "recommendations": final[:3]}

    def _generate_forensic_explanation(self, persona, trace, threat, transfer_count=0):
        objective = {"FASTEST": "Transit time", "SAFEST": "Risk-weighted time", "BALANCED": "Time, freight cost and risk"}[persona]
        eta, cost = sum(trace["eta"].values()), sum(trace["cost"].values())
        return (f"{objective} optimized. Estimated transit plus buffers: {eta:.1f}h; "
                f"freight and transfer cost: ${cost:.2f}; {transfer_count} transfers. "
                f"Model buffer: {trace['eta']['model']:.1f}h; additional scenario buffer: "
                f"{trace['eta']['scenario']:.1f}h; peak threat: {threat:.0%}.")
