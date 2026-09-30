# Verification — 30 September 2026

## Automated checks

- **22 tests passed** with `python -m unittest discover -s tests -v`.
- Supplied `risk_model.pkl` loaded with scikit-learn **1.8.0**, its training version.
- Model loss is `quantile` and alpha is **0.85**.
- **Frontend production build passed** with `npm run build` from `frontend`.
  Vite emitted a non-blocking warning for an existing bundle larger than 500 kB.
- `git diff --check` passed.
- Local FastAPI server startup and `/api/status` HTTP response checked.

## Original bug reproduction

Shanghai → Rotterdam, strict SEA, original implementation:

| Scenario | Fastest ETA | Sum of ETA audit components | Total cost | Sum of cost audit components |
|---|---:|---:|---:|---:|
| Normal | 413.9 h | 413.9 h | $2,565.33 | $2,565.33 |
| SUEZ_BLOCK | 653.9 h | 893.9 h | $2,565.33 | $2,600.16 |

The original audit counted 240 scenario hours twice and reported a surcharge that was not
charged in `total_cost`. `CARFFilter(.8, 'A vessel blocked the canal.', 'sea')` returned 0.0.

## Improved offline results

The trained model is enabled; transformer/RSS access is disabled for the reproducible demo.

| Scenario | Fastest ETA | Sum of ETA audit components | Total cost | Route behaviour |
|---|---:|---:|---:|---|
| Normal | 561.7 h | 561.7 h | $2,751.07 | Via Suez Canal |
| SUEZ_BLOCK | 832.3 h | 832.3 h | $4,259.39 | Avoids Suez; via Durban and Algeciras |

API tests also verify the dashboard's equivalent hub selections:
`HUB-SHANGHAI` → `HUB-ROTTERDAM`. A subsequent normal request returns to the normal route.

A controlled small graph test verifies that increasing the model's delay alone changes
route selection. It does not rely on changed response labels or a scenario override.

## What the tests cover

- Correct positive NLP threshold and per-chunk threat pooling.
- Relevant and irrelevant news for sea, air, road and rail, including punctuation/plurals.
- Real model loading, batch/single prediction parity and calibrated bounds.
- Visible fallback estimates and unknown-hub encoding disclosures.
- Model-driven rerouting and matching-mode scenario application.
- No repeated scenario delay on transfer edges, no double-counted model/scenario overlap.
- ETA and cost reconciliation, constraints, invalid inputs, and excluded endpoints.
- API status accurately reflects model state; API validation and scenario discovery work.
- Canonical Suez routing and normal-request isolation.

## Verification limits

- The downloaded headless-browser archive was corrupted, so visual/browser interaction
  testing could not be completed. JSX compilation and the API calls used by the UI passed.
- Transformer weights were not installed. NLP math is tested with controlled similarity
  matrices; live transformer embeddings and news relevance quality remain unverified.
- Windows launch scripts are provided but were not executed on Windows. Their backend
  Python commands and frontend npm commands were verified on Linux.
- No GitHub push, upstream pull request, or competition submission was performed.
