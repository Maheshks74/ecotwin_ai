
"""Streamlit Interactive Digital Twin Dashboard for EcoTwin."""
from __future__ import annotations

from pathlib import Path
import os
import sys

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ecotwin.env.ecotwin_env import EcoTwinEnv
from ecotwin.controllers.fixed import FixedController
from ecotwin.controllers.rule_based import RuleBasedController
from ecotwin.controllers.ppo_train import PPOAgent, train_ppo
from ecotwin.controllers.ppo_pref import PPOPrefAgent
from ecotwin.llm.client import GeminiClient
from ecotwin.llm.advisor import LLMAdvisor
from ecotwin.llm.instruction_parser import InstructionParser
from ecotwin.llm.logger import FloorLogger
from ecotwin.data.scenarios import ScenarioManager


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EcoTwin | Industrial Digital Twin & RL Energy Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# INITIALIZE LLM CLIENT
# ============================================================

@st.cache_resource
def get_llm_client():
    return GeminiClient()


llm_client = get_llm_client()
advisor = LLMAdvisor(client=llm_client)
instruction_parser = InstructionParser(client=llm_client)
scenario_mgr = ScenarioManager()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏭 EcoTwin Platform")

if llm_client.is_mock_mode:
    st.sidebar.warning("⚠️ **Mode**: Mock LLM (Offline)")
    st.sidebar.caption(
        "Using deterministic canned responses. "
        "Add GEMINI_API_KEY in .env for Live Gemini."
    )
else:
    st.sidebar.success(f"🟢 **Mode**: {llm_client.badge}")

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Controls")

available_scenarios = scenario_mgr.list_scenarios()

selected_scenario = st.sidebar.selectbox(
    "Manufacturing Scenario",
    available_scenarios,
    index=0
)

controller_options = [
    "Rule-Based Heuristic",
    "Fixed Schedule",
    "PPO Hand-Reward",
    "PPO Preference-Reward"
]

selected_controller_name = st.sidebar.selectbox(
    "Active Controller",
    controller_options,
    index=0
)

simulation_seed = st.sidebar.number_input(
    "Random Seed",
    value=42,
    min_value=1,
    max_value=999999
)


# ============================================================
# CONTROLLER SELECTION
# ============================================================

def get_controller(name: str):

    if name == "Fixed Schedule":
        return FixedController()

    elif name == "Rule-Based Heuristic":
        return RuleBasedController()

    elif name == "PPO Hand-Reward":
        model_path = "models/ppo_hand_model.zip"

        return PPOAgent(
            model_path=model_path if Path(model_path).exists() else None
        )

    elif name == "PPO Preference-Reward":
        model_path = "models/ppo_pref_model.zip"

        return PPOPrefAgent(
            model_path=model_path if Path(model_path).exists() else None
        )

    return RuleBasedController()


# ============================================================
# PPO EVALUATION FUNCTION
# ============================================================

def evaluate_ppo_model(
    model,
    scenario_name: str,
    seed: int,
    reward_weights: dict
):
    """
    Evaluate a trained PPO policy on a fixed scenario and seed.
    Returns actual simulation KPIs.
    """

    env = EcoTwinEnv(
        scenario_name=scenario_name,
        reward_weights=reward_weights,
        include_weather_forecast=True
    )

    logger = FloorLogger()

    try:
        obs, info = env.reset(
            seed=int(seed),
            options={"scenario": scenario_name}
        )

        terminated = False
        truncated = False

        episode_reward = 0.0
        final_info = {}

        while not (terminated or truncated):

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            step_idx = env.sim.current_step

            price = env.tariff_mgr.get_price(step_idx)

            outside_temp = env.weather_mgr.get_temperature(step_idx)

            obs, reward, terminated, truncated, step_info = env.step(action)

            episode_reward += float(reward)

            logger.log_step(
                env.last_result,
                action,
                outside_temp,
                price,
                reward,
                scenario_name=scenario_name
            )

            final_info = step_info

    finally:
        env.close()

    telemetry = logger.to_dataframe()

    if telemetry.empty:
        raise RuntimeError(
            "Evaluation produced no telemetry rows."
        )

    return {
        "telemetry": telemetry,

        "episode_reward": episode_reward,

        "cost_inr": float(
            telemetry["cumulative_cost_inr"].iloc[-1]
        ),

        "energy_kwh": float(
            telemetry["cumulative_energy_kwh"].iloc[-1]
        ),

        "peak_kw": float(
            telemetry["peak_power_kw"].max()
        ),

        "completed_jobs": int(
            telemetry["completed_jobs"].iloc[-1]
        ),

        "quota": int(
            telemetry["quota"].iloc[-1]
        ),

        "comfort_violation": float(
            telemetry["total_comfort_violation"].sum()
        ),

        "final_info": final_info
    }


# ============================================================
# REWARD COMPARISON FUNCTION
# ============================================================

def reward_comparison_frame(before: dict, after: dict):

    keys = [
        "energy_cost_weight",
        "demand_charge_weight",
        "comfort_violation_weight",
        "quota_penalty_weight",
        "late_penalty_weight"
    ]

    return pd.DataFrame({

        "Reward component": keys,

        "Before": [
            float(before[k]) for k in keys
        ],

        "After": [
            float(after[k]) for k in keys
        ]
    })


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "EcoTwin: Industrial Digital Twin with RL Energy Optimization"
)

st.markdown(
    "Physics-grounded manufacturing floor digital twin with "
    "peak demand trimming, comfort management, and LLM operational co-pilot."
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "🏭 Floor Telemetry & Replay",
    "📊 Controller Benchmark",
    "🤖 AI Operations Advisor & What-If",
    "⚙️ Reward Tuning & Directives"
])


# ============================================================
# TAB 1: LIVE SIMULATION AND TELEMETRY
# ============================================================

with tabs[0]:

    st.subheader(
        f"Simulation Replay: {selected_controller_name} "
        f"on {selected_scenario.replace('_', ' ').title()}"
    )

    if st.button(
        "▶️ Run 24-Hour Digital Twin Simulation",
        use_container_width=True
    ):

        with st.spinner(
            "Simulating 96 time-steps (15-min intervals)..."
        ):

            env = EcoTwinEnv(
                scenario_name=selected_scenario
            )

            ctrl = get_controller(
                selected_controller_name
            )

            logger = FloorLogger()

            try:

                obs, info = env.reset(
                    seed=simulation_seed,
                    options={"scenario": selected_scenario}
                )

                term = False
                truncated = False
                step_info = {}

                while not (term or truncated):

                    act = ctrl.act(obs)

                    step_idx = env.sim.current_step

                    price = env.tariff_mgr.get_price(step_idx)

                    out_t = env.weather_mgr.get_temperature(step_idx)

                    obs, reward, term, truncated, step_info = env.step(act)

                    logger.log_step(
                        env.last_result,
                        act,
                        out_t,
                        price,
                        reward,
                        scenario_name=selected_scenario
                    )

            finally:
                env.close()

            st.session_state["last_telemetry"] = logger.to_dataframe()

            st.session_state["last_info"] = step_info

    if "last_telemetry" in st.session_state:

        df = st.session_state["last_telemetry"]

        last_info = st.session_state.get(
            "last_info",
            {}
        )

        # KPIs

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

        kpi1.metric(
            "Electricity Cost",
            f"₹ {df['cumulative_cost_inr'].iloc[-1]:,.2f}"
        )

        kpi2.metric(
            "Total Energy",
            f"{df['cumulative_energy_kwh'].iloc[-1]:,.1f} kWh"
        )

        kpi3.metric(
            "Peak Power",
            f"{df['peak_power_kw'].max():,.1f} kW"
        )

        kpi4.metric(
            "Jobs Done",
            f"{df['completed_jobs'].iloc[-1]} / {df['quota'].iloc[-1]}"
        )

        kpi5.metric(
            "Comfort Violations",
            f"{df['total_comfort_violation'].sum():.1f} °C·h"
        )

        col_left, col_right = st.columns(2)

        # POWER CHART

        with col_left:

            fig_p = go.Figure()

            fig_p.add_trace(
                go.Scatter(
                    x=df["hour"],
                    y=df["total_power_kw"],
                    name="Total Floor Power",
                    line=dict(color="#3182CE", width=2)
                )
            )

            fig_p.add_trace(
                go.Scatter(
                    x=df["hour"],
                    y=df["machines_power_kw"],
                    name="Machines Power",
                    line=dict(color="#38A169", dash="dot")
                )
            )

            fig_p.add_trace(
                go.Scatter(
                    x=df["hour"],
                    y=df["hvac_power_kw"],
                    name="HVAC Power",
                    line=dict(color="#D69E2E", dash="dash")
                )
            )

            fig_p.add_vrect(
                x0=18.0,
                x1=22.0,
                fillcolor="red",
                opacity=0.15,
                layer="below",
                line_width=0,
                annotation_text="Peak Tariff (18-22h)"
            )

            fig_p.update_layout(
                title="Floor Power Profile (24 Hours)",
                xaxis_title="Hour of Day",
                yaxis_title="Power (kW)",
                height=380
            )

            st.plotly_chart(
                fig_p,
                use_container_width=True
            )

        # TEMPERATURE CHART

        with col_right:

            fig_t = go.Figure()

            fig_t.add_trace(
                go.Scatter(
                    x=df["hour"],
                    y=df["outside_temp_c"],
                    name="Ambient Outside",
                    line=dict(color="red", dash="dash")
                )
            )

            fig_t.add_trace(
                go.Scatter(
                    x=df["hour"],
                    y=df["zone_1_temp_c"],
                    name="Zone 1 (Machining)",
                    line=dict(color="#3182CE", width=2)
                )
            )

            fig_t.add_trace(
                go.Scatter(
                    x=df["hour"],
                    y=df["zone_2_temp_c"],
                    name="Zone 2 (Assembly)",
                    line=dict(color="#38A169", width=2)
                )
            )

            fig_t.add_hrect(
                y0=22.0,
                y1=26.0,
                fillcolor="green",
                opacity=0.15,
                layer="below",
                line_width=0,
                annotation_text="Comfort Band (22-26°C)"
            )

            fig_t.update_layout(
                title="Thermal Zones vs Outside Ambient",
                xaxis_title="Hour of Day",
                yaxis_title="Temperature (°C)",
                height=380
            )

            st.plotly_chart(
                fig_t,
                use_container_width=True
            )

        # MACHINE STATES

        st.subheader("Machine States Across Shifts")

        machine_cols = [
            f"state_{mid}"
            for mid in ["M1", "M2", "M3", "M4", "M5", "M6"]
            if f"state_{mid}" in df.columns
        ]

        if machine_cols:

            df_states = df[
                ["hour"] + machine_cols
            ].set_index("hour")

            mapping = {
                "IDLE": 0,
                "HELD": 1,
                "PROCESSING": 2
            }

            df_numeric = df_states.replace(mapping)

            fig_states = px.imshow(
                df_numeric.T,
                labels=dict(
                    x="Hour of Day",
                    y="Machine",
                    color="State"
                ),
                x=df["hour"],
                color_continuous_scale=[
                    (0.0, "#E2E8F0"),
                    (0.5, "#ECC94B"),
                    (1.0, "#48BB78")
                ],
                aspect="auto"
            )

            fig_states.update_layout(
                height=250,
                coloraxis_colorbar=dict(
                    title="State",
                    tickvals=[0, 1, 2],
                    ticktext=["IDLE", "HELD", "PROCESSING"]
                )
            )

            st.plotly_chart(
                fig_states,
                use_container_width=True
            )

    else:

        st.info(
            "Click '▶️ Run 24-Hour Digital Twin Simulation' "
            "above to simulate and visualize telemetry."
        )


# ============================================================
# TAB 2: CONTROLLER BENCHMARK
# ============================================================

with tabs[1]:

    st.subheader(
        "Benchmark Comparison Across All Four Controllers"
    )

    results_csv = Path(
        "evaluation/results_summary.csv"
    )

    if results_csv.exists():

        df_bench = pd.read_csv(results_csv)

        st.dataframe(
            df_bench,
            use_container_width=True
        )

        # COST COMPARISON

        fig_bar = px.bar(
            df_bench,
            x="scenario",
            y="cost_inr",
            color="controller",
            barmode="group",
            title="Electricity Cost by Controller and Scenario",
            labels={
                "cost_inr": "Total Cost (INR)",
                "scenario": "Scenario"
            }
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

        # PEAK POWER COMPARISON

        fig_peak = px.bar(
            df_bench,
            x="scenario",
            y="peak_kw",
            color="controller",
            barmode="group",
            title="Peak Demand (kW) by Controller and Scenario",
            labels={
                "peak_kw": "Peak Demand (kW)",
                "scenario": "Scenario"
            }
        )

        st.plotly_chart(
            fig_peak,
            use_container_width=True
        )

        # ABLATION STUDY

        ablation_csv = Path(
            "evaluation/ablation_results.csv"
        )

        if ablation_csv.exists():

            st.subheader(
                "Ablation Study: Weather Forecast Impact"
            )

            df_abl = pd.read_csv(ablation_csv)

            st.dataframe(
                df_abl,
                use_container_width=True
            )

    else:

        st.warning(
            "Benchmark results not yet generated. "
            "Run the full evaluation in Phase 8 or execute "
            "`python evaluation/run_eval.py`."
        )


# ============================================================
# TAB 3: AI OPERATIONS ADVISOR AND WHAT-IF
# ============================================================

with tabs[2]:

    st.subheader(
        "🤖 Operations Advisor & What-If Exploration"
    )

    advisor_col1, advisor_col2 = st.columns([1, 1])

    # TELEMETRY ADVISOR

    with advisor_col1:

        st.markdown(
            "### Telemetry Grounded Explanations"
        )

        user_question = st.text_input(
            "Ask a question regarding operational decisions:",
            value="Why was Job 14 moved to 2 AM?",
            key="advisor_question"
        )

        if st.button(
            "Ask Advisor",
            key="btn_advisor"
        ):

            if "last_telemetry" in st.session_state:

                telemetry_data = st.session_state[
                    "last_telemetry"
                ]

            else:

                telemetry_data = pd.DataFrame([
                    {
                        "step": 8,
                        "hour": 2.0,
                        "price_inr_per_kwh": 4.5,
                        "outside_temp_c": 22.0,
                        "total_power_kw": 40.0,
                        "machines_power_kw": 30.0,
                        "hvac_power_kw": 10.0,
                        "zone_1_temp_c": 24.0,
                        "zone_2_temp_c": 24.0,
                        "min_slack": 20.0,
                        "completed_jobs": 5
                    }
                ])

            with st.spinner(
                "Analyzing digital twin telemetry..."
            ):

                answer = advisor.explain(
                    user_question,
                    telemetry_data
                )

                st.markdown(
                    f"**Advisor Response** ({llm_client.badge}):"
                )

                st.info(answer)

    # WHAT-IF ANALYSIS

    with advisor_col2:

        st.markdown(
            "### Digital Twin What-If Studio"
        )

        whatif_query = st.text_input(
            "Test a hypothetical operational scenario:",
            value="What if temperature setpoint is relaxed by 1 degree to save electricity?",
            key="whatif_query"
        )

        if st.button(
            "Execute What-If Simulation",
            key="btn_whatif"
        ):

            with st.spinner(
                "Running comparative digital twin episodes..."
            ):

                whatif_res = advisor.run_what_if_analysis(
                    whatif_query,
                    base_scenario=selected_scenario
                )

                st.markdown(
                    "**Comparative What-If Analysis**:"
                )

                st.success(
                    whatif_res["summary"]
                )

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Cost Impact",
                    f"₹ {whatif_res['cost_delta_inr']:+.2f}",
                    f"{whatif_res['savings_pct']:+.1f}%"
                )

                m2.metric(
                    "Energy Impact",
                    f"{whatif_res['kwh_delta']:+.1f} kWh"
                )

                m3.metric(
                    "Peak Power Impact",
                    f"{whatif_res['peak_delta_kw']:+.1f} kW"
                )


# ============================================================
# TAB 4: NATURAL-LANGUAGE REWARD COMPILER + PPO ADAPTATION
# ============================================================

with tabs[3]:

    st.subheader(
        "⚙️ Natural-Language Reward Compiler + PPO Adaptation"
    )

    st.markdown(
        "Compile a manager directive into reward weights, "
        "train baseline and adapted PPO policies, "
        "then evaluate both on the same scenario and seed."
    )

    # DEFAULT REWARD WEIGHTS

    default_weights = {
        key: float(value)
        for key, value in EcoTwinEnv.DEFAULT_REWARD_WEIGHTS.items()
    }

    col_directive, col_setup = st.columns([1.15, 0.85])

    # --------------------------------------------------------
    # 1. NATURAL LANGUAGE COMPILER
    # --------------------------------------------------------

    with col_directive:

        st.markdown(
            "### 1. Natural-language directive"
        )

        directive = st.text_area(
            "Manager instruction",
            value=st.session_state.get(
                "manager_directive",
                "Customer batch X is highly urgent. "
                "Avoid missed deadlines even if electricity is expensive."
            ),
            height=110,
            key="manager_directive_input"
        )

        if st.button(
            "Compile directive with Gemini",
            key="btn_compile",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Compiling directive into validated reward weights..."
                ):

                    parsed_weights, reasoning = (
                        instruction_parser.parse_instruction(directive)
                    )

                # IMPORTANT FIX:
                # Validate against InstructionParser's schema,
                # rather than assuming the environment dictionary
                # is the source of the parser schema.

                required = set(
                    InstructionParser.ALLOWED_BOUNDS.keys()
                )

                if not isinstance(parsed_weights, dict):

                    raise TypeError(
                        "InstructionParser must return a dictionary "
                        "of reward weights. "
                        f"Received: {type(parsed_weights).__name__}"
                    )

                missing = required - set(
                    parsed_weights.keys()
                )

                if missing:

                    raise ValueError(
                        "Missing reward-weight fields: "
                        f"{sorted(missing)}. "
                        "Received fields: "
                        f"{list(parsed_weights.keys())}"
                    )

                # Validate values and bounds.

                compiled_weights = {}

                for key, (low, high) in (
                    InstructionParser.ALLOWED_BOUNDS.items()
                ):

                    value = parsed_weights[key]

                    if isinstance(value, bool):

                        raise ValueError(
                            f"{key} must be numeric, not boolean."
                        )

                    try:

                        value = float(value)

                    except (TypeError, ValueError):

                        raise ValueError(
                            f"{key} must contain a numeric value."
                        )

                    if not np.isfinite(value):

                        raise ValueError(
                            f"{key} must be a finite number."
                        )

                    if not low <= value <= high:

                        raise ValueError(
                            f"{key}={value} is outside the "
                            f"allowed range [{low}, {high}]."
                        )

                    compiled_weights[key] = value

                # Check that parser and environment keys match.

                environment_keys = set(EcoTwinEnv.DEFAULT_REWARD_WEIGHTS.keys())
                parser_keys = set(InstructionParser.ALLOWED_BOUNDS.keys())

                # Parser keys must exist in the environment.
                # Environment may have additional technical parameters.
                if not parser_keys.issubset(environment_keys):
                    raise ValueError(
                        f"Parser contains unsupported reward keys: "
                        f"{sorted(parser_keys - environment_keys)}"
                    )

                # Save compiled weights.
                # Merge any environment‑only reward keys (e.g., reward_scale) using defaults.
                for extra_key in environment_keys - parser_keys:
                    compiled_weights[extra_key] = EcoTwinEnv.DEFAULT_REWARD_WEIGHTS[extra_key]

                st.session_state[
                    "compiled_reward_weights"
                ] = compiled_weights

                st.session_state[
                    "compiled_reasoning"
                ] = str(reasoning)

                st.success(
                    "Directive compiled successfully. "
                    "Review the weights before training."
                )

            except Exception as exc:

                st.error(
                    f"Directive compilation failed: {exc}"
                )

                st.exception(exc)

        # DISPLAY COMPILED RESULT

        compiled = st.session_state.get(
            "compiled_reward_weights"
        )

        if compiled:

            st.markdown(
                "**Compiler reasoning**"
            )

            st.write(
                st.session_state.get(
                    "compiled_reasoning",
                    ""
                )
            )

            st.markdown(
                "**Compiled reward weights**"
            )

            st.dataframe(
                reward_comparison_frame(
                    default_weights,
                    compiled
                ),
                hide_index=True,
                use_container_width=True
            )

        else:

            st.info(
                "Compile a directive to prepare "
                "the adapted reward configuration."
            )

    # --------------------------------------------------------
    # 2. EXPERIMENT SETTINGS
    # --------------------------------------------------------

    with col_setup:

        st.markdown(
            "### 2. Experiment settings"
        )

        timesteps = st.number_input(
            "Training timesteps per PPO model",
            min_value=1000,
            max_value=200000,
            value=15000,
            step=1000,
            help=(
                "Both baseline and adapted PPO receive "
                "this same training budget."
            )
        )

        eval_seed = st.number_input(
            "Evaluation seed",
            min_value=1,
            max_value=999999,
            value=int(simulation_seed)
        )

        st.caption(
            f"Evaluation scenario: **{selected_scenario}**"
        )

        st.caption(
            "Both policies are evaluated deterministically "
            "with the same scenario and seed."
        )

        st.warning(
            "Training two PPO models can take several minutes. "
            "The adapted model is retrained from a fresh "
            "initialization using the compiled reward weights."
        )

    # --------------------------------------------------------
    # 3. TRAIN BOTH PPO MODELS
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "### 3. Train and compare"
    )

    can_run = bool(
        st.session_state.get(
            "compiled_reward_weights"
        )
    )

    if st.button(
        "🚀 Train baseline + adapted PPO and run comparison",
        key="btn_train_compare",
        type="primary",
        use_container_width=True,
        disabled=not can_run
    ):

        adapted_weights = dict(
            st.session_state["compiled_reward_weights"]
        )

        baseline_path = (
            "models/directive_experiment_baseline.zip"
        )

        adapted_path = (
            "models/directive_experiment_adapted.zip"
        )

        experiment_seed = int(eval_seed)

        try:

            Path("models").mkdir(
                parents=True,
                exist_ok=True
            )

            progress = st.progress(
                0,
                text="Preparing baseline PPO training..."
            )

            status = st.empty()

            # BASELINE TRAINING

            status.info(
                "Training baseline PPO with the environment's "
                "default reward weights..."
            )

            baseline_model = train_ppo(
                total_timesteps=int(timesteps),
                seed=experiment_seed,
                save_path=baseline_path,
                tensorboard_log="runs/directive_experiment_baseline",
                checkpoint_freq=max(
                    1000,
                    int(timesteps) // 3
                ),
                randomize_scenarios=True,
                reward_weights=default_weights,
                verbose=0
            )

            progress.progress(
                45,
                text="Baseline PPO training complete."
            )

            # ADAPTED TRAINING

            status.info(
                "Training adapted PPO from scratch "
                "with Gemini-compiled reward weights..."
            )

            adapted_model = train_ppo(
                total_timesteps=int(timesteps),
                seed=experiment_seed,
                save_path=adapted_path,
                tensorboard_log="runs/directive_experiment_adapted",
                checkpoint_freq=max(
                    1000,
                    int(timesteps) // 3
                ),
                randomize_scenarios=True,
                reward_weights=adapted_weights,
                verbose=0
            )

            progress.progress(
                85,
                text="Both models trained. "
                     "Evaluating on matched conditions..."
            )

            # EVALUATE BASELINE

            baseline_result = evaluate_ppo_model(
                baseline_model,
                selected_scenario,
                experiment_seed,
                default_weights
            )

            # EVALUATE ADAPTED

            adapted_result = evaluate_ppo_model(
                adapted_model,
                selected_scenario,
                experiment_seed,
                adapted_weights
            )

            progress.progress(
                100,
                text="Experiment completed."
            )

            status.success(
                "Both PPO models trained and evaluated successfully."
            )

            # STORE RESULTS

            st.session_state[
                "directive_experiment"
            ] = {

                "scenario": selected_scenario,

                "seed": experiment_seed,

                "timesteps": int(timesteps),

                "baseline_weights": default_weights,

                "adapted_weights": adapted_weights,

                "baseline": baseline_result,

                "adapted": adapted_result,

                "baseline_model_path": baseline_path,

                "adapted_model_path": adapted_path
            }

        except Exception as exc:

            st.error(
                f"Experiment failed. No comparison was saved: {exc}"
            )

            st.exception(exc)

    # --------------------------------------------------------
    # 4. DISPLAY BEFORE AND AFTER RESULTS
    # --------------------------------------------------------

    experiment = st.session_state.get(
        "directive_experiment"
    )

    if experiment:

        st.divider()

        st.markdown(
            "### 4. Before-and-after results"
        )

        st.caption(
            f"Scenario: {experiment['scenario']} · "
            f"Evaluation seed: {experiment['seed']} · "
            f"Training budget: "
            f"{experiment['timesteps']:,} timesteps per model"
        )

        before = experiment["baseline"]

        after = experiment["adapted"]

        # RESULT TABLE

        metric_rows = [

            (
                "Episode reward",
                before["episode_reward"],
                after["episode_reward"]
            ),

            (
                "Electricity cost (₹)",
                before["cost_inr"],
                after["cost_inr"]
            ),

            (
                "Energy (kWh)",
                before["energy_kwh"],
                after["energy_kwh"]
            ),

            (
                "Peak power (kW)",
                before["peak_kw"],
                after["peak_kw"]
            ),

            (
                "Completed jobs",
                before["completed_jobs"],
                after["completed_jobs"]
            ),

            (
                "Comfort violation (°C·h)",
                before["comfort_violation"],
                after["comfort_violation"]
            )
        ]

        result_df = pd.DataFrame([

            {
                "Metric": name,
                "Baseline PPO": old,
                "Directive PPO": new
            }

            for name, old, new in metric_rows

        ])

        st.dataframe(
            result_df,
            hide_index=True,
            use_container_width=True
        )

        # KPI COMPARISON

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Baseline cost",
            f"₹{before['cost_inr']:,.2f}"
        )

        k2.metric(
            "Directive cost",
            f"₹{after['cost_inr']:,.2f}",
            f"{before['cost_inr'] - after['cost_inr']:+,.2f} vs baseline"
        )

        k3.metric(
            "Baseline jobs",
            f"{before['completed_jobs']} / {before['quota']}"
        )

        k4.metric(
            "Directive jobs",
            f"{after['completed_jobs']} / {after['quota']}",
            f"{after['completed_jobs'] - before['completed_jobs']:+d} jobs"
        )

        # CHART DATA

        chart_df = pd.DataFrame({

            "Controller": [
                "Baseline PPO",
                "Directive PPO"
            ],

            "Electricity cost (₹)": [
                before["cost_inr"],
                after["cost_inr"]
            ],

            "Energy (kWh)": [
                before["energy_kwh"],
                after["energy_kwh"]
            ],

            "Peak power (kW)": [
                before["peak_kw"],
                after["peak_kw"]
            ],

            "Completed jobs": [
                before["completed_jobs"],
                after["completed_jobs"]
            ]
        })

        c1, c2 = st.columns(2)

        # COST AND ENERGY CHARTS

        with c1:

            fig_cost = px.bar(
                chart_df,
                x="Controller",
                y="Electricity cost (₹)",
                title="Electricity cost: baseline vs directive"
            )

            st.plotly_chart(
                fig_cost,
                use_container_width=True
            )

            fig_energy = px.bar(
                chart_df,
                x="Controller",
                y="Energy (kWh)",
                title="Energy consumption: baseline vs directive"
            )

            st.plotly_chart(
                fig_energy,
                use_container_width=True
            )

        # JOBS AND PEAK POWER CHARTS

        with c2:

            fig_jobs = px.bar(
                chart_df,
                x="Controller",
                y="Completed jobs",
                title="Production completed: baseline vs directive"
            )

            st.plotly_chart(
                fig_jobs,
                use_container_width=True
            )

            fig_peak = px.bar(
                chart_df,
                x="Controller",
                y="Peak power (kW)",
                title="Peak demand: baseline vs directive"
            )

            st.plotly_chart(
                fig_peak,
                use_container_width=True
            )

        # REWARD CONFIGURATION COMPARISON

        st.markdown(
            "**Reward configuration used for this experiment**"
        )

        st.dataframe(
            reward_comparison_frame(
                experiment["baseline_weights"],
                experiment["adapted_weights"]
            ),
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "These are single-seed evaluation results, not statistical "
            "proof of improvement. Repeat with multiple evaluation "
            "seeds before making research claims."
        )

        # CSV EXPORT

        st.download_button(
            "Download comparison CSV",
            data=result_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="ecotwin_directive_ppo_comparison.csv",
            mime="text/csv"
        )
