import subprocess
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "install_refresh_cron.sh")
NIGHTLY = os.path.join(ROOT, "scripts", "install_nightly_validation_cron.sh")


def run(cmd):
    p = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout.decode()


def test_print_refresh_cron_contains_schedule():
    out = run(f"bash {SCRIPT} --print")
    # Expect a schedule line and the refresh script path (weekly refresh runs at 03:00 UTC by default)
    assert "/scripts/refresh_fec_rag.sh" in out
    assert "0 3" in out


def test_print_nightly_cron_contains_tz_and_schedule():
    out = run(f"bash {SCRIPT} --print-nightly")
    # The nightly script prints CRON_TZ and the schedule
    assert "CRON_TZ=America/Los_Angeles" in out
    assert "0 0" in out


def test_print_all_contains_both_entries():
    out = run(f"bash {SCRIPT} --print-all")
    assert "/scripts/refresh_fec_rag.sh" in out
    assert "CRON_TZ=America/Los_Angeles" in out
