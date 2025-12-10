"""Configuration settings for the nurse scheduling project."""

from pathlib import Path

N_NURSES = 20
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SHIFTS = ["M", "A", "N"]  # Morning, Afternoon, Night

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
