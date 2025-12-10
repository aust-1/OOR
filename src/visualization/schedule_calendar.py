"""Generate a calendar visualization of the nurse scheduling solution."""

import matplotlib.pyplot as plt
import pandas as pd

from config import DAYS, N_NURSES, OUTPUT_DIR

names = [f"Nurse_{i + 1}" for i in range(N_NURSES)]
schedule = pd.read_csv(OUTPUT_DIR / "solution.csv")

df = schedule[schedule.assigned == 1]

fig, ax = plt.subplots(figsize=(11, 6))
for i, (nurse_id, name, day, shift, _, _) in df.iterrows():
    ax.text(
        DAYS.index(day),
        names.index(name),
        shift,
        ha="center",
        va="center",
        fontsize=8,
    )

ax.set_xticks(range(len(DAYS)))
ax.set_xticklabels(DAYS)
ax.set_yticks(range(N_NURSES))
ax.set_yticklabels(names)
plt.title("Weekly Nurse Roster")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roster_table.png", dpi=300)
