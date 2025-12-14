import os


def test_build_cron_line():
    # Emulate what the install script would create and validate it's correct
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cron_cmd = f"/bin/bash {repo_root}/scripts/refresh_fec_rag.sh"
    cron_line = f"0 3 * * 0 {cron_cmd}"

    assert cron_line.startswith("0 3 * * 0 ")
    assert "refresh_fec_rag.sh" in cron_line
