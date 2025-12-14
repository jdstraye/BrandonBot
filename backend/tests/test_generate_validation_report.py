import os
import json
from scripts.generate_validation_report import generate_report


def test_generate_report_tmp(tmp_path):
    runs_dir = tmp_path / 'runs'
    out_dir = tmp_path / 'report'
    runs_dir.mkdir()

    # create two fake runs
    for i in range(2):
        data = {
            'timestamp': f'2025-12-14T0{i}:00:00',
            'results': [
                {'score_clarity': 4.0, 'score_empathy': 4.0, 'score_accuracy': 4.0, 'score_engagement': 4.0, 'score_tone': 4.0, 'score_alignment': 4.0}
            ]
        }
        with open(runs_dir / f'validation_run_test_{i}.json', 'w') as f:
            json.dump(data, f)

    generate_report(str(runs_dir), str(out_dir))
    assert (out_dir / 'index.html').exists()
