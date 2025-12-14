#!/usr/bin/env python3
"""Generate an HTML validation report from JSON run artifacts.

Produces `validation_report/index.html` and a PNG plot of average judge scores over time.
"""
import os
import json
from datetime import datetime
import glob
import argparse

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_runs(dirpath):
    runs = []
    for p in sorted(glob.glob(os.path.join(dirpath, 'validation_run_*.json'))):
        with open(p) as f:
            data = json.load(f)
            runs.append((p, data))
    return runs


def compute_average_score(result):
    # result is dict form of TestResult; use judge fields if present
    try:
        scores = [result.get('score_clarity', 0), result.get('score_empathy', 0), result.get('score_accuracy', 0), result.get('score_engagement', 0), result.get('score_tone', 0), result.get('score_alignment', 0)]
        return sum(scores) / len(scores)
    except Exception:
        return 0


def generate_report(runs_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    runs = load_runs(runs_dir)

    series = []
    for path, run in runs:
        ts = run.get('timestamp') or os.path.basename(path).replace('validation_run_','').replace('.json','')
        avg_scores = []
        for r in run.get('results', []):
            avg_scores.append(compute_average_score(r))
        avg = sum(avg_scores)/len(avg_scores) if avg_scores else 0
        series.append((ts, avg, path))

    # Plot
    if plt:
        times = [datetime.fromisoformat(s[0]) for s in series]
        vals = [s[1] for s in series]
        plt.figure(figsize=(8,3))
        plt.plot(times, vals, marker='o')
        plt.title('Average Judge Score Over Time')
        plt.ylabel('Average Judge Score')
        plt.xlabel('Timestamp')
        plt.grid(True)
        plot_path = os.path.join(out_dir, 'avg_judge_score.png')
        plt.tight_layout()
        plt.savefig(plot_path)
    else:
        plot_path = None

    # Simple HTML
    html = '<html><head><title>Validation Report</title></head><body>'
    html += '<h1>Validation Report</h1>'
    html += f'<p>Runs analyzed: {len(series)}</p>'
    if plot_path:
        html += f'<img src="{os.path.basename(plot_path)}" alt="avg plot" />'

    html += '<table border="1"><tr><th>timestamp</th><th>avg_score</th><th>artifact</th></tr>'
    for ts, avg, path in series:
        html += f'<tr><td>{ts}</td><td>{avg:.2f}</td><td>{os.path.basename(path)}</td></tr>'
    html += '</table>'
    html += '</body></html>'

    with open(os.path.join(out_dir, 'index.html'), 'w') as f:
        f.write(html)

    print('Report written to', out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', default='validation_runs')
    parser.add_argument('--out-dir', default='validation_report')
    args = parser.parse_args()
    generate_report(args.runs_dir, args.out_dir)


if __name__ == '__main__':
    main()
