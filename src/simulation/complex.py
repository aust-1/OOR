"""Generate complex nurse rostering simulation data. Prefers, skill levels, varied coverage."""

import random

import numpy as np
import pandas as pd

from config import DAYS, N_NURSES, OUTPUT_DIR, SHIFTS

# region nurses.csv
rng = np.random.default_rng(42)

nurse_ids = list(range(N_NURSES))
names = [f"Nurse_{i + 1}" for i in nurse_ids]

skill_levels = rng.choice(
    ["junior", "senior", "icu"], size=N_NURSES, p=[0.5, 0.35, 0.15]
)
contracts = rng.choice(["full_time", "part_time"], size=N_NURSES, p=[0.7, 0.3])

base_abs = rng.uniform(0.01, 0.04, size=N_NURSES)
for i, lvl in enumerate(skill_levels):
    if lvl == "icu":
        base_abs[i] *= 0.7
absence_prob = np.round(base_abs, 3)

nurses_df = pd.DataFrame(
    {
        "nurse_id": nurse_ids,
        "name": names,
        "skill_level": skill_levels,
        "contract": contracts,
        "absence_prob": absence_prob,
    }
)

nurses_df.to_csv(OUTPUT_DIR / "nurses.csv", index=False)
# endregion

# region coverage.csv
rows = []
for day in DAYS:
    for shift in SHIFTS:
        if day in ["Sat", "Sun"]:
            required_total = 6 if shift != "N" else 3
        else:
            required_total = 5 if shift != "N" else 3

        if shift == "N":
            required_senior = 1
            required_icu = 1
        else:
            required_senior = 1
            required_icu = 0 if day not in ["Sat", "Sun"] else 1

        rows.append(
            {
                "day": day,
                "shift": shift,
                "required_total": required_total,
                "required_senior": required_senior,
                "required_icu": required_icu,
            }
        )

coverage_df = pd.DataFrame(rows)
coverage_df.to_csv(OUTPUT_DIR / "coverage.csv", index=False)
# endregion

# region preferences.csv
pref_rows = []

for nurse_id in nurse_ids:
    n_prefs = rng.integers(2, 6)

    for _ in range(n_prefs):
        day = random.choice(DAYS)
        shift = random.choice(SHIFTS)

        pref_type = "avoid" if rng.random() < 0.6 else "want"

        weight = int(rng.integers(1, 6))

        pref_rows.append(
            {
                "nurse_id": nurse_id,
                "day": day,
                "shift": shift,
                "preference_type": pref_type,
                "weight": weight,
            }
        )

preferences_df = pd.DataFrame(pref_rows).drop_duplicates()
preferences_df.to_csv(OUTPUT_DIR / "preferences.csv", index=False)
# endregion
