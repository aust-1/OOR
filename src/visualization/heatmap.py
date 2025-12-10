"""Generate a heatmap visualization of the nurse scheduling solution."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import DAYS, OUTPUT_DIR, SHIFTS

schedule = pd.read_csv(OUTPUT_DIR / "solution.csv")

schedule["day"] = pd.Categorical(schedule["day"], categories=DAYS, ordered=True)
schedule["shift"] = pd.Categorical(schedule["shift"], categories=SHIFTS, ordered=True)

pivot = schedule.pivot_table(
    index="nurse_id", columns=["day", "shift"], values="assigned", fill_value=0
).sort_index(axis=1, level=[0, 1])
plt.figure(figsize=(10, 8))
sns.heatmap(
    pivot,
    cmap="Greens",
    cbar=False,
    linewidths=0.5,
    linecolor="#e5e5e5",
)
plt.title("Generated Nurse Schedule - Heatmap Representation")
plt.tight_layout()
plt.savefig("heatmap_schedule.png", dpi=300)
