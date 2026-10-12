"""Warm the Statcast cache for the given seasons: python fetch_seasons.py 2023 2024"""
import sys
from baseball_data import statcast_pitches

for y in map(int, sys.argv[1:]):
    df = statcast_pitches(f"{y}-03-20", f"{y}-10-05")
    print(y, df.shape, flush=True)
