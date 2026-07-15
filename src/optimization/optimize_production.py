# =============================================================================
# optimize_production.py — Production Allocation Optimizer
#
# Uses Linear Programming (PuLP) to determine the optimal production
# rate for each well, maximizing total field output while respecting
# individual well health constraints AND shared infrastructure limits.
# =============================================================================

import os
import sys
import json
import pandas as pd
import pulp

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_and_solve_lp(
    wells_df: pd.DataFrame,
    infra_limits: dict
) -> dict:
    """
    Formulates and solves the production allocation problem as
    a Linear Program.

    PuLP works by letting you define variables, an objective, and
    constraints in near-mathematical notation, then hands the
    problem to a solver (CBC, bundled with PuLP by default) which
    applies the Simplex algorithm - a well-established method that
    systematically explores the "corners" of the feasible region
    (the space of all valid solutions) to find the mathematically
    optimal point, rather than guessing or searching randomly.
    """
    logger.info("Building Linear Program...")

    # 1. Create the problem - we want to MAXIMIZE total production
    prob = pulp.LpProblem("Production_Allocation", pulp.LpMaximize)

    # 2. Decision variables - one continuous variable per well,
    # bounded between 0 and that well's SAFE (health-adjusted) capacity.
    # This upper bound is itself a constraint, built directly into
    # the variable definition rather than a separate constraint line.
    production_vars = {
        row["well_id"]: pulp.LpVariable(
            row["well_id"], lowBound=0, upBound=row["safe_capacity_bpd"]
        )
        for _, row in wells_df.iterrows()
    }

    # 3. Objective function - maximize the SUM of all wells' production
    prob += pulp.lpSum(production_vars.values()), "Total_Field_Production"

    # 4. Shared infrastructure constraints - total production across
    # ALL wells cannot exceed pipeline or compressor capacity
    prob += (
        pulp.lpSum(production_vars.values()) <= infra_limits["pipeline_capacity_bpd"],
        "Pipeline_Capacity_Limit"
    )
    prob += (
        pulp.lpSum(production_vars.values()) <= infra_limits["compressor_capacity_bpd"],
        "Compressor_Capacity_Limit"
    )

    # Fairness constraint: every well must produce AT LEAST 30% of its
    # safe capacity. Without this, the solver's pure "maximize total"
    # objective has no incentive to distribute production fairly - it
    # will happily zero out some wells entirely if that reaches the
    # same total faster. Real operations require minimum baseline
    # production from every active well for reservoir management
    # and business reasons the raw objective function doesn't capture.
    for _, row in wells_df.iterrows():
        min_required = row["safe_capacity_bpd"] * 0.3
        prob += (
            production_vars[row["well_id"]] >= min_required,
            f"Min_Fairness_{row['well_id']}"
        )

    logger.info(f"Problem formulated: {len(production_vars)} variables, "
                f"3 constraint groups + fairness minimums")

    # 5. Solve - PuLP calls the CBC solver under the hood
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    logger.info(f"Solver status: {status}")

    results = []
    for well_id, var in production_vars.items():
        results.append({
            "well_id": well_id,
            "allocated_production_bpd": round(var.varValue, 1)
        })

    results_df = pd.DataFrame(results)
    total_allocated = results_df["allocated_production_bpd"].sum()

    logger.info(f"Total optimal production: {total_allocated:.1f} bpd")

    return {
        "status": status,
        "results_df": results_df,
        "total_production": total_allocated,
        "objective_value": pulp.value(prob.objective)
    }


def analyze_binding_constraints(
    wells_df: pd.DataFrame,
    infra_limits: dict,
    solution: dict
) -> list:
    """
    Identifies which constraints are "binding" - i.e., fully used up,
    meaning they're actively LIMITING the solution. This is the
    optimization equivalent of SHAP: it explains WHY the solver
    couldn't allocate more, in plain terms a planner can act on.
    """
    logger.info("Analyzing binding constraints...")

    binding = []
    total = solution["total_production"]

    if abs(total - infra_limits["pipeline_capacity_bpd"]) < 0.5:
        binding.append(
            f"Pipeline capacity ({infra_limits['pipeline_capacity_bpd']} bpd) "
            f"is FULLY UTILIZED - this is the limiting factor for total output"
        )

    if abs(total - infra_limits["compressor_capacity_bpd"]) < 0.5:
        binding.append(
            f"Compressor capacity ({infra_limits['compressor_capacity_bpd']} bpd) "
            f"is FULLY UTILIZED - this is the limiting factor for total output"
        )

    merged = wells_df.merge(solution["results_df"], on="well_id")
    at_max_wells = merged[
        abs(merged["allocated_production_bpd"] - merged["safe_capacity_bpd"]) < 0.5
    ]

    for _, row in at_max_wells.iterrows():
        binding.append(
            f"{row['well_id']} is producing at its FULL safe capacity "
            f"({row['safe_capacity_bpd']:.0f} bpd) - {row['health_status']}"
        )

    for msg in binding:
        logger.info(f"  → {msg}")

    return binding


def run_optimization():
    logger.info("=" * 60)
    logger.info("Production Optimization Copilot")
    logger.info("=" * 60)

    wells_df = pd.read_csv("data/optimization/wells.csv")
    with open("data/optimization/infrastructure_limits.json") as f:
        infra_limits = json.load(f)

    solution = build_and_solve_lp(wells_df, infra_limits)
    binding_constraints = analyze_binding_constraints(wells_df, infra_limits, solution)

    merged = wells_df.merge(solution["results_df"], on="well_id")
    merged["utilization_pct"] = (
        merged["allocated_production_bpd"] / merged["safe_capacity_bpd"] * 100
    ).round(1)

    os.makedirs("data/optimization", exist_ok=True)
    merged.to_csv("data/optimization/optimal_allocation.csv", index=False)

    with open("data/optimization/binding_constraints.json", "w") as f:
        json.dump(binding_constraints, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Optimization complete: {solution['total_production']:.1f} bpd "
                f"(status: {solution['status']})")
    logger.info("=" * 60)

    return merged, binding_constraints


if __name__ == "__main__":
    merged, binding = run_optimization()
    print(merged.to_string(index=False))
