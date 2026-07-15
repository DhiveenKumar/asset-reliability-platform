# =============================================================================
# whatif_simulator.py — What-If Scenario Simulation
#
# Re-runs the LP solver under modified constraints to answer
# planning questions: "what if pipeline capacity increases?",
# "what if a specific well degrades further?"
# =============================================================================

import os
import sys
import json
import pandas as pd
import copy

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger
from src.optimization.optimize_production import build_and_solve_lp, analyze_binding_constraints

logger = get_logger(__name__)


def simulate_pipeline_expansion(
    wells_df: pd.DataFrame,
    infra_limits: dict,
    additional_capacity: float
) -> dict:
    """
    Scenario: What if we invest in expanding pipeline capacity?
    Answers the business question "how much extra production would
    X additional bpd of pipeline capacity actually unlock?" - directly
    useful for capital investment decisions.
    """
    logger.info(f"\\n--- SCENARIO: Pipeline +{additional_capacity} bpd ---")

    modified_limits = copy.deepcopy(infra_limits)
    modified_limits["pipeline_capacity_bpd"] += additional_capacity

    solution = build_and_solve_lp(wells_df, modified_limits)

    logger.info(f"New total production: {solution['total_production']:.1f} bpd")

    return solution


def simulate_well_degradation(
    wells_df: pd.DataFrame,
    infra_limits: dict,
    well_id: str,
    new_derate_factor: float
) -> dict:
    """
    Scenario: What if a specific well's health degrades further?
    Answers "if WELL-X's condition worsens, how does that ripple
    through the whole field's optimal allocation?" - useful for
    proactive planning around assets RULSense flags as declining.
    """
    logger.info(f"\\n--- SCENARIO: {well_id} degrades to "
                f"{new_derate_factor:.0%} of max capacity ---")

    modified_wells = wells_df.copy()
    mask = modified_wells["well_id"] == well_id
    original_max = modified_wells.loc[mask, "max_capacity_bpd"].values[0]
    new_safe_capacity = original_max * new_derate_factor

    modified_wells.loc[mask, "safe_capacity_bpd"] = new_safe_capacity
    modified_wells.loc[mask, "health_status"] = "SIMULATED - Further Degraded"

    solution = build_and_solve_lp(modified_wells, infra_limits)

    logger.info(f"New total production: {solution['total_production']:.1f} bpd")

    return solution


def compare_scenarios(baseline: dict, scenario: dict, scenario_name: str) -> dict:
    """
    Quantifies the IMPACT of a scenario versus the baseline -
    this is what actually gets reported to a decision-maker,
    not just the raw numbers.
    """
    delta = scenario["total_production"] - baseline["total_production"]
    pct_change = (delta / baseline["total_production"]) * 100

    comparison = {
        "scenario": scenario_name,
        "baseline_production": round(baseline["total_production"], 1),
        "scenario_production": round(scenario["total_production"], 1),
        "delta_bpd": round(delta, 1),
        "pct_change": round(pct_change, 2)
    }

    logger.info(
        f"IMPACT: {delta:+.1f} bpd ({pct_change:+.1f}%) vs baseline"
    )

    return comparison


def run_all_scenarios():
    logger.info("=" * 60)
    logger.info("What-If Scenario Analysis")
    logger.info("=" * 60)

    wells_df = pd.read_csv("data/optimization/wells.csv")
    with open("data/optimization/infrastructure_limits.json") as f:
        infra_limits = json.load(f)

    # Re-apply fairness constraint by using the same solver function
    # (imported directly, so behavior is guaranteed consistent)
    baseline = build_and_solve_lp(wells_df, infra_limits)
    logger.info(f"BASELINE production: {baseline['total_production']:.1f} bpd")

    scenarios = []

    pipeline_scenario = simulate_pipeline_expansion(wells_df, infra_limits, 500)
    scenarios.append(compare_scenarios(baseline, pipeline_scenario, "Pipeline +500 bpd"))

    degradation_scenario = simulate_well_degradation(wells_df, infra_limits, "WELL-009", 0.5)
    scenarios.append(compare_scenarios(baseline, degradation_scenario, "WELL-009 degrades 50%"))

    compressor_scenario_limits = copy.deepcopy(infra_limits)
    compressor_scenario_limits["compressor_capacity_bpd"] += 300
    compressor_scenario = build_and_solve_lp(wells_df, compressor_scenario_limits)
    scenarios.append(compare_scenarios(baseline, compressor_scenario, "Compressor +300 bpd"))

    scenarios_df = pd.DataFrame(scenarios)

    os.makedirs("data/optimization", exist_ok=True)
    scenarios_df.to_csv("data/optimization/whatif_scenarios.csv", index=False)

    logger.info("=" * 60)
    logger.info("Scenario Comparison Summary")
    logger.info("=" * 60)
    logger.info("\\n" + scenarios_df.to_string(index=False))

    return scenarios_df


if __name__ == "__main__":
    run_all_scenarios()
