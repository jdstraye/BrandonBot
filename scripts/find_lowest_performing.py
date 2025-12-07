#!/usr/bin/env python3
import csv
from collections import defaultdict

# This is the file that actually has your data
CSV_PATH = "/home/runner/workspace/backend/validation/results/validation_results_latest.csv"

print("Loading real validation results...\n")

dialogs = defaultdict(list)
scores  = {}

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        tid = row["Test_ID"].strip()

        # Save every turn
        dialogs[tid].append({
            "turn": row["Turn"],
            "user": row["User_Prompt"] or "",
            "bot":  row["Bot_Response"] or "(empty)"
        })

        # The evaluated rows are the ones where Score_Clarity is present AND > 0
        # In your real file, these start appearing after the PQ-* tests
        try:
            clarity = float(row["Score_Clarity"])
            if clarity > 0:                # ← THIS IS THE KEY LINE
                scores[tid] = {
                    "Clarity":    clarity,
                    "Empathy":    float(row["Score_Empathy"] or 0),
                    "Accuracy":   float(row["Score_Accuracy"] or 0),
                    "Engagement": float(row["Score_Engagement"] or 0),
                    "Tone":       float(row["Score_Tone"] or 0),
                    "Alignment":  float(row["Score_Alignment"] or 0),
                }
        except:
            continue

if not scores:
    print("No evaluated conversations (Score_Clarity > 0) found in this file.")
    print("Make sure the evaluation run has finished and real scores are present.")
    exit(1)

# Find worst in each category
categories = ["Clarity", "Empathy", "Accuracy", "Engagement", "Tone", "Alignment"]

print("="*90)
print("LOWEST PERFORMING CONVERSATIONS (real scored ones only)")
print("="*90)

for cat in categories:
    worst_tid   = min(scores, key=lambda x: scores[x][cat])
    worst_score = scores[worst_tid][cat]

    print(f"\nScore_{cat}: {worst_score} → Test_ID: {worst_tid}\n")
    print("Full conversation:")
    print("-" * 70)
    for t in sorted(dialogs[worst_tid], key=lambda x: int(x["turn"])):
        print(f"Turn {t['turn']}")
        if t["user"]:
            print(f"User: {t['user']}")
        print(f"Bot : {t['bot']}")
        print()
    print("="*90)