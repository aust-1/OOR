"""Basic simulation data generation. 20 nurses, 7 days, 3 shifts per day."""

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

nurses = [f"Nurse_{i + 1}" for i in range(20)]
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
shifts = ["M", "A", "N"]  # Morning / Afternoon / Night

coverage = pd.DataFrame(
    {
        "Day": np.repeat(days, len(shifts)),
        "Shift": shifts * len(days),
        "Required": [5, 5, 3] * len(days),
    }
)

coverage.to_csv(OUTPUT_DIR / "coverage.csv", index=False)
pd.DataFrame({"Nurses": nurses}).to_csv(OUTPUT_DIR / "nurses.csv", index=False)
