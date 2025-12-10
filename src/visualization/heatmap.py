"""Generate a heatmap visualization of the nurse scheduling solution."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

schedule = pd.read_csv("solution.csv")

pivot = schedule.pivot_table(
    index="Nurse", columns=["Day", "Shift"], values="Assigned", fill_value=0
)
plt.figure(figsize=(10, 8))
sns.heatmap(pivot, cmap="Greens", cbar=False)
plt.title("Generated Nurse Schedule - Heatmap Representation")
plt.tight_layout()
plt.savefig("heatmap_schedule.png", dpi=300)
