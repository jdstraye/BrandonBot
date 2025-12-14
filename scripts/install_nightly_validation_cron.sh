#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CMD="/usr/bin/env python3 $ROOT_DIR/scripts/run_full_validation.py --output-dir $ROOT_DIR/validation_runs && /usr/bin/env python3 $ROOT_DIR/scripts/generate_validation_report.py --runs-dir $ROOT_DIR/validation_runs --out-dir $ROOT_DIR/validation_report"
# Run nightly at 00:00 (midnight) Pacific Time (America/Los_Angeles)
# Use CRON_TZ to ensure the schedule is interpreted in America/Los_Angeles
CRON_TZ_LINE="CRON_TZ=America/Los_Angeles"
CRON_SCHEDULE="0 0 * * *"
CRON_LINE="$CRON_TZ_LINE\n$CRON_SCHEDULE $CMD"

if [ "$#" -gt 0 ] && [ "$1" = "--print" ]; then
  echo "$CRON_LINE"
  exit 0
fi

CRON_NOW=$(crontab -l 2>/dev/null || true)

# If the command is present, ensure we update the schedule to the desired one
if echo "$CRON_NOW" | grep -F -q "$CMD"; then
  # Remove any existing line that contains the command and any existing CRON_TZ for this job
  # Use '|' as delimiter to avoid conflicts with '/' in the command path
  echo "$CRON_NOW" | sed "\|$CMD|d" | sed "/^CRON_TZ=America\/Los_Angeles$/d" > /tmp/crontab.tmp
  (cat /tmp/crontab.tmp; echo "# Nightly validation report (added by BrandonBot)"; echo -e "$CRON_LINE") | crontab -
  rm -f /tmp/crontab.tmp
  echo "Updated nightly validation cron (00:00 America/Los_Angeles)."
  exit 0
fi

# Append lines, preserving existing crontab
(crontab -l 2>/dev/null || true; echo "# Nightly validation report (added by BrandonBot)") | { cat; echo -e "$CRON_LINE"; } | crontab -
echo "Installed nightly validation cron (00:00 America/Los_Angeles)."
