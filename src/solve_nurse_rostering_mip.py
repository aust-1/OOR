"""
Solve a nurse rostering problem from three CSV files using OR-Tools linear solver.

Required files (in DATA_DIR):
- nurses.csv:
    nurse_id,name,skill_level,contract,absence_prob
- coverage.csv:
    day,shift,required_total,required_senior,required_icu
- preferences.csv:
    nurse_id,day,shift,preference_type,weight

Output:
- solution.csv:
    nurse_id,name,day,shift,assigned,skill_level
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from ortools.linear_solver import pywraplp

from config import OUTPUT_DIR
from structs import CoverageRequirement, Nurse, Preference


def load_data(
    data_dir: Path,
) -> Tuple[
    List[Nurse],
    List[CoverageRequirement],
    List[Preference],
    List[str],
    List[str],
]:
    """Load nurses, coverage and preferences from CSV files."""
    nurses_df = pd.read_csv(data_dir / "nurses.csv")
    coverage_df = pd.read_csv(data_dir / "coverage.csv")
    prefs_df = pd.read_csv(data_dir / "preferences.csv")

    nurses = [
        Nurse(
            nurse_id=int(row["nurse_id"]),
            name=str(row["name"]),
            skill_level=str(row["skill_level"]),
            contract=str(row["contract"]),
            absence_prob=float(row["absence_prob"]),
        )
        for _, row in nurses_df.iterrows()
    ]

    coverage = [
        CoverageRequirement(
            day=str(row["day"]),
            shift=str(row["shift"]),
            required_total=int(row["required_total"]),
            required_senior=int(row["required_senior"]),
            required_icu=int(row["required_icu"]),
        )
        for _, row in coverage_df.iterrows()
    ]

    preferences = [
        Preference(
            nurse_id=int(row["nurse_id"]),
            day=str(row["day"]),
            shift=str(row["shift"]),
            preference_type=str(row["preference_type"]).lower(),
            weight=int(row["weight"]),
        )
        for _, row in prefs_df.iterrows()
    ]

    days = list(dict.fromkeys(coverage_df["day"].tolist()))
    shifts = list(dict.fromkeys(coverage_df["shift"].tolist()))

    return nurses, coverage, preferences, days, shifts


def build_and_solve(
    nurses: List[Nurse],
    coverage: List[CoverageRequirement],
    preferences: List[Preference],
    days: List[str],
    shifts: List[str],
    time_limit_sec: int = 60,
) -> pd.DataFrame:
    """
    Build the MIP model, solve it, and return a schedule DataFrame.

    The objective is to minimize penalties from violated preferences.
    """
    solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        raise RuntimeError("Failed to create CBC solver.")
    solver.SetNumThreads(8)
    solver.SetTimeLimit(time_limit_sec * 1000)

    nurse_ids = [n.nurse_id for n in nurses]

    nurse_by_id: Dict[int, Nurse] = {n.nurse_id: n for n in nurses}

    cov_map: Dict[Tuple[str, str], CoverageRequirement] = {
        (c.day, c.shift): c for c in coverage
    }

    x: Dict[Tuple[int, str, str], pywraplp.Variable] = {}
    for nid in nurse_ids:
        for d in days:
            for s in shifts:
                x[(nid, d, s)] = solver.BoolVar(f"x_n{nid}_{d}_{s}")

    for nid in nurse_ids:
        for d in days:
            solver.Add(solver.Sum(x[(nid, d, s)] for s in shifts) <= 1)

    senior_like = {n.nurse_id for n in nurses if n.skill_level in {"senior", "icu"}}
    icu_only = {n.nurse_id for n in nurses if n.skill_level == "icu"}

    for d in days:
        for s in shifts:
            cov = cov_map[(d, s)]

            solver.Add(
                solver.Sum(x[(nid, d, s)] for nid in nurse_ids) >= cov.required_total
            )

            if cov.required_senior > 0:
                solver.Add(
                    solver.Sum(x[(nid, d, s)] for nid in senior_like)
                    >= cov.required_senior
                )

            if cov.required_icu > 0:
                solver.Add(
                    solver.Sum(x[(nid, d, s)] for nid in icu_only) >= cov.required_icu
                )

    penalty_vars: List[Tuple[pywraplp.Variable, int]] = []
    for i, pref in enumerate(preferences):
        key = (pref.nurse_id, pref.day, pref.shift)
        if key not in x:
            continue

        assign_var = x[key]

        if pref.preference_type == "avoid":
            penalty = solver.BoolVar(f"penalty_pref_{i}")
            solver.Add(penalty == assign_var)
            penalty_vars.append((penalty, pref.weight))
        elif pref.preference_type == "want":
            penalty = solver.BoolVar(f"penalty_pref_{i}")
            solver.Add(penalty + assign_var == 1)
            penalty_vars.append((penalty, pref.weight))
        else:
            continue

    objective = solver.Objective()
    for penalty, weight in penalty_vars:
        objective.SetCoefficient(penalty, weight)
    objective.SetMinimization()

    start = time.perf_counter()
    status = solver.Solve()
    runtime = time.perf_counter() - start

    feasible_statuses = {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}
    if status not in feasible_statuses:
        raise RuntimeError(f"No feasible solution found (status={status}).")

    rows = []
    for nid in nurse_ids:
        nurse = nurse_by_id[nid]
        for d in days:
            for s in shifts:
                assigned = int(round(x[(nid, d, s)].solution_value()))
                rows.append(
                    {
                        "nurse_id": nid,
                        "name": nurse.name,
                        "day": d,
                        "shift": s,
                        "assigned": assigned,
                        "skill_level": nurse.skill_level,
                    }
                )

    schedule_df = pd.DataFrame(rows)

    print("Objective:", objective.Value())
    print("Runtime (s):", runtime)
    print("Status:", "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE")

    return schedule_df


def main() -> None:
    nurses, coverage, preferences, days, shifts = load_data(OUTPUT_DIR)

    schedule_df = build_and_solve(
        nurses=nurses,
        coverage=coverage,
        preferences=preferences,
        days=days,
        shifts=shifts,
        time_limit_sec=60,
    )

    schedule_df.to_csv(OUTPUT_DIR / "solution.csv", index=False)
    print(f"Solution saved to: {(OUTPUT_DIR / 'solution.csv').resolve()}")

    assigned_counts = (
        schedule_df[schedule_df["assigned"] == 1].groupby("nurse_id")["assigned"].sum()
    )
    print("\nAssigned shifts per nurse:")
    print(assigned_counts.describe())


if __name__ == "__main__":
    main()
