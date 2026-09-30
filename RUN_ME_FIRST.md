# Supplychainer — Avinash's improved challenge project

## What this does

This is the original TatHack starter project with verified routing fixes and one
complete feature: the supplied trained delay model now influences route selection,
and the dashboard shows the prediction evidence.

## Run on Windows

1. Install **Python 3.12** (include the Python launcher) and **Node.js LTS** if they
   are not already installed. These are the two programs required to run this project.
2. Extract the ZIP completely. Do not run files from inside the ZIP preview.
3. Open the extracted `Supplychainer_Improved` folder.
4. Double-click `START_BACKEND.bat`. On first launch, it installs the Python packages.
   Wait until it says the API is running on `http://127.0.0.1:8000`.
5. Double-click `START_FRONTEND.bat`. On first launch, it installs the frontend packages.
6. Open **http://localhost:5173** in your browser. Leave both command windows open.

Internet is needed for the first dependency installation. The scenario demonstration
then runs without downloading NLP weights or fetching news. The trained sklearn model
still runs in demo mode. This is not a live-news demonstration.

If a command window shows an error, copy that text or send a screenshot to your assistant.
If ports 8000 or 5173 are already in use, close the previous project windows first.

## What to demonstrate to the judges

1. Search for **Shanghai** in Origin Hub and select **Shanghai Global Distribution Center**.
2. Search for **Rotterdam** in Destination Hub and select **Rotterdam Distripark**.
3. Choose **SEA** transport and **STRICT** routing policy.
4. Keep **Operational Normal** and click **GENERATE STRATEGIC ROUTE OPTIONS**.
5. Note the route through Suez, the ETA, and the model buffer in the right audit panel.
6. Choose **Suez Canal Blockage** and generate again.
7. Show that the route avoids Suez and travels through Durban, with a longer transit time.
8. Explain that a route avoiding the affected hub can have low threat even while the
   scenario is active. The scenario name alone does not mean every route has threat 1.0.
9. Switch back to normal and generate again. The disruption must not leak into that request.

Offline tested city-equivalent values for this graph/model: normal fastest route about
**561.7 hours**, blocked fastest sea route about **832.3 hours**. These are prototype
estimates, not shipping promises; live NLP initialization can change the baseline.

## Your short explanation

“I connected the existing delay model to the live routing algorithm. Previously, the
model was loaded but its predictions did not influence the route. I also fixed reversed
news scoring and mode relevance checks. The engine now uses the same delay numbers for
route selection and the audit display. Our tests check route changes and reconciled totals,
not just whether the program runs.”

## What was fixed

- Strong disaster-vs-safe NLP scores were discarded; the threshold now retains positive threats.
- Relevant maritime/aviation news was filtered out; the relevance filter now handles all four modes.
- The ML model was unused by live routing; batch predictions now feed every transit-edge weight.
- ETA audit added scenario delay twice; transit, transfers, model buffers and additional scenario
  buffers are now separate, additive components.
- Audit claimed a 10% scenario surcharge that was absent from the actual cost. It now reports
  only the freight and transfer charges actually included in the total.
- Unsupported cities silently became the first training city. They now use explicit generic
  categories, with the substitution disclosed in prediction evidence.
- Inference failure used to return a zero buffer. It now uses a clearly labeled operational prior.
- Scenario configuration used shared mutable state. Route requests now use independent snapshots.
- Comparisons such as “396% savings” were not based on an alternative route. Explanations now
  state measured totals without invented savings.
- Model files now resolve relative to the code, and torch is optional for offline routing.
- Startup scores four fallback news texts once instead of repeating NLP for every edge.

## Tests and development

From the project root after starting the backend once:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-demo.txt
DEMO_MODE=true uvicorn backend.main:app --host 127.0.0.1 --port 8000
# In another terminal:
cd frontend
npm ci
npm run dev
```

Build the frontend with `cd frontend` followed by `npm run build`.

For the original NLP initialization, install compatible `torch` and `sentence-transformers`
packages in the backend environment and unset `DEMO_MODE`. This path is optional and was
not verified with downloaded transformer weights in this change. It initializes the
existing fallback news text, not a newly implemented live RSS routing pipeline.

## Honest limitations

- The original data builder creates generated samples from hardcoded benchmark anchors;
  this is not a newly validated dataset of 50,000 observed shipment logs. The statistical
  quality of the original model was not independently validated here.
- Per-leg p85 calibrated buffers are summed. Their sum is **not** a proven route-level
  85th-percentile guarantee, and a p85 quantile is not an absolute worst case.
- A scenario delay is treated as a minimum buffer for the affected incoming transit edge.
  It is combined with the severity-conditioned model using `max(model, scenario)` so the
  same disruption is not charged twice. Transfers do not receive the scenario again.
- Scenarios currently apply to incoming edges of matching transport mode. Modeling a
  disruption at the departure hub and correlated whole-corridor closures needs more work.
- Graph distances, costs, and transfers retain the starter's approximations; some transfer
  paths may favor an extra handoff under the SAFEST heuristic.
- Cost ceilings and maximum lead-time limits filter each persona's chosen path. This is not
  an exhaustive constrained-shortest-path solver.
- Route priority and budget sensitivity are still starter inputs without a fully defined
  optimization policy. No claim is made that those features were completed.
- This is a local prototype, without production authentication or persistent route history.

Read `VERIFICATION.md` for the tested results and `changes.patch` for the exact code changes.
