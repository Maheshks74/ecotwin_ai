# EcoTwin Project Progress

## Environment & Setup
- **Anaconda Environment**: `base` (Python 3.13.9, Anaconda 64-bit on Windows)
- **Execution Mode**: `conda run -n base python ...` / `C:\Users\mahes\anaconda3\python.exe`
- **Device**: CPU-only PyTorch (no discrete GPU detected / requested)
- **Packages Installed**:
  - `torch==2.15.0+cpu` (CPU-only wheel)
  - `tensorboard==2.21.0`
  - `stable-baselines3==2.9.2`
  - `google-genai==2.34.0`
  - `dulwich==1.2.20` (pure-Python Git tracking)
  - Existing packages reused: `simpy 4.1.2`, `gymnasium 1.4.0`, `pandas 2.3.3`, `numpy 2.3.5`, `pyyaml 6.0.3`, `streamlit 1.51.0`, `plotly 6.3.0`, `matplotlib 3.10.6`, `pytest 8.4.2`, `requests 2.32.5`, `python-dotenv`, `pyarrow 21.0.0`
- **Known Conflicts & Mitigations**:
  - Intel OpenMP duplicate runtime (`libiomp5md.dll`) between Anaconda base and PyTorch wheel: resolved via setting `KMP_DUPLICATE_LIB_OK=TRUE` at package startup and in project entry points.

## Phase Tracking

| Phase | Description | Status | Test Status | Commits / Notes |
|---|---|---|---|---|
| Setup | Environment setup, Git init, requirements.txt, .gitignore, .env.example | Complete | All imports verified | Initial commit |
| Phase 1 | Digital twin (SimPy) 6 machines, 2 HVAC zones, RC model, 15-min / 96 steps | Complete | 4/4 pytest passed, smoke passed | 6 machines, 2 HVAC zones, energy units verified |
| Phase 2 | Data pipeline (Tariff TOU/IEX, Weather Open-Meteo & synthetic forecast, Scenarios) | Complete | 6/6 pytest passed, smoke passed | TOU slabs, IEX CSV loader, demand charge, weather forecast uncertainty growth, 5 scenarios |
| Phase 3 | Gymnasium environment (Normalized Box obs, MultiDiscrete action, Reward, check_env) | Complete | 4/4 pytest passed (check_env, 1000 steps, quota proof, weights), smoke passed | Box(58,), MultiDiscrete([2,2,2,2,2,2,5,5]), lazy-vs-active quota penalty verified |
| Phase 4 | Baseline controllers (Fixed schedule, Rule-based deferral & pre-cooling) | Complete | 3/3 pytest passed, smoke passed | Standardized act(obs)->action, verified pre-cooling and heavy job peak deferral |
| Phase 5 | PPO training (SB3 PPO, randomized scenarios, Tensorboard logging, checkpoints) | Complete | 2/2 pytest passed, smoke passed | PPOAgent wrapper, checkpoints, multi-seed support, model save/load verified |
| Phase 6 | Preference learning (Simulated manager, trajectory pairs, Bradley-Terry reward model, ppo_pref) | Complete | 3/3 pytest passed, smoke passed | Latent manager utilities, pair generation, Bradley-Terry BCE loss, PPO preference wrapper |
| Phase 7 | LLM advisor (Gemini client with Mock mode, logger, advisor, what-if, instruction parser) | Complete | 5/5 pytest passed, smoke passed | Mock LLM badge, factual retrieval, 'not enough information' fallback, schema validation |
| Phase 8 | Evaluation (run_eval over 4 controllers, CSV metrics, Plotly/Matplotlib figures) | Complete | 3/3 pytest passed, smoke passed | Multi-seed evaluation, cost/peak metrics, weather ablation, Matplotlib and Plotly figures |
| Phase 9 | Dashboard (Streamlit app with replay, comparison, advisor chat, reward sliders, mock badge) | Complete | Streamlit AppTest passed, smoke passed | 4 tabs (Replay, Benchmark, Advisor/What-If, Directives), Mock/Live badge |

## Benchmark Evaluation Highlights (Real Evaluation Numbers)

- **Peak Trimming**: PPO Hand-Reward reduced peak power demand from **125.33 kW** (Fixed) down to **65.14 kW** (**48.0% reduction** on Summer Day) and **71.69 kW** vs **129.21 kW** (**44.5% reduction** on Rush Order).
- **Cost Savings**: PPO Hand-Reward achieved **18.4% cost savings** on Rush Order (₹ 5,399.33 vs ₹ 6,619.95) and **16.2% savings** on Summer Day (₹ 5,325.80 vs ₹ 6,355.30).
- **Quota Fulfillment**: 100% quota achieved across all benchmark scenarios and seeds (40/30 jobs on standard scenarios, 55/45 on heavy quota, 48/32 on rush order).
- **Weather Forecast Ablation**: On Summer Day, PPO with Weather Forecast saved **₹ 866.47 (14.0%)** over ablated PPO without Weather Forecast, and trimmed peak by **16.5 kW (20.2%)**.
- **Preference Learning**: Bradley-Terry neural reward model achieved **80.0% accuracy** on held-out manager preference pairs and learned to penalize night shifts and temperature excursions.

## Test Summary
- **Total Pytest Tests**: 31 / 31 passed (`tests/test_twin.py`, `tests/test_data.py`, `tests/test_env.py`, `tests/test_controllers.py`, `tests/test_ppo.py`, `tests/test_preference.py`, `tests/test_llm.py`, `tests/test_eval.py`, `tests/test_dashboard.py`)
- **End-to-End Smoke Test**: `scripts/smoke.py` passes all 9 phases in ~15 seconds offline.
- **Git State**: Clean commit history tracking each phase completion.

