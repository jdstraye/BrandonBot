#!/usr/bin/env python3
import csv
import sqlite3
from pathlib import Path

# Update these paths to match your project layout
CSV_FILE = "backend/validation/results/validation_results_latest.csv"
DB_FILE   = "backend/validation/results/validation_debug.db"

# Verify files exist
for p in (CSV_FILE, DB_FILE):
    if not Path(p).exists():
        raise FileNotFoundError(f"File not found: {p}")

print("Loading scored conversations from CSV...")
scored = {}  # test_id → {Clarity: x, Empathy: y, ...}

with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tid = row["Test_ID"].strip()
        try:
            clarity = float(row["Score_Clarity"] or 0)
            if clarity > 0:                               # only evaluated rows
                scored[tid] = {
                    "Clarity":    clarity,
                    "Empathy":    float(row["Score_Empathy"] or 0),
                    "Accuracy":   float(row["Score_Accuracy"] or 0),
                    "Engagement": float(row["Score_Engagement"] or 0),
                    "Tone":       float(row["Score_Tone"] or 0),
                    "Alignment":  float(row["Score_Alignment"] or 0),
                }
        except ValueError:
            continue

if not scored:
    raise ValueError("No scored conversations found in the CSV!")

# Find lowest score per category
categories = ["Clarity", "Empathy", "Accuracy", "Engagement", "Tone", "Alignment"]
lowest = {}

for cat in categories:
    min_score = min(info[cat] for info in scored.values())
    test_id   = next(tid for tid, info in scored.items() if info[cat] == min_score)
    lowest[cat] = (test_id, min_score)

# Connect to DB and pull full dialogs
conn = sqlite3.connect(DB_FILE)
cur  = conn.cursor()

# Confirm the real table name (should print "dialog_turns")
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables in DB:", [r[0] for r in cur.fetchall()])

print("\n" + "="*80)
print("LOWEST PERFORMING CONVERSATIONS BY CATEGORY")
print("="*80)

for cat, (test_id, score) in lowest.items():
    print(f"\nLowest Score_{cat}: {score}  →  Test_ID: {test_id}\n")

    cur.execute("""
        SELECT turn, user_prompt, bot_response
        FROM dialog_turns
        WHERE test_id = ?
        ORDER BY turn
    """, (test_id,))

    rows = cur.fetchall()

    if not rows:
        print("  No dialog found in database for this Test_ID")
        continue

    for turn, user, bot in rows:
        print(f"Turn {turn}")
        print(f"User: {user}")
        if bot is None:
            bot = "(no response / tool call only)"
        print(f"Bot : {bot}")
        print("-" * 60)

    print("\n" + "="*80)

conn.close()
print("\nFinished!")