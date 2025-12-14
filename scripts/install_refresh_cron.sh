#!/usr/bin/env bash
set -euo pipefail

# Install or show a crontab entry to run the FEC RAG refresh weekly and keep logs.
# Usage:
#   ./scripts/install_refresh_cron.sh --install             # install FEC RAG refresh into user's crontab
#   ./scripts/install_refresh_cron.sh --print               # print the FEC RAG refresh crontab entry
#   ./scripts/install_refresh_cron.sh --install-nightly     # only install nightly validation cron
#   ./scripts/install_refresh_cron.sh --install-all         # install both refresh + nightly (single-step)
#   ./scripts/install_refresh_cron.sh --print-nightly       # print the nightly validation crontab entry

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CRON_CMD="/bin/bash $ROOT_DIR/scripts/refresh_fec_rag.sh"
# Run every Sunday at 03:00 UTC
CRON_SCHEDULE="0 3 * * 0"
CRON_LINE="$CRON_SCHEDULE $CRON_CMD"

print_usage() {
  echo "Usage: $0 [--install|--print|--install-nightly|--install-all|--print-nightly|--print-all]"
  exit 2
}

if [ "$1" = "--print-all" ]; then
  echo "# FEC RAG refresh"
  echo "$CRON_LINE"
  echo "# Nightly validation"
  bash "$ROOT_DIR/scripts/install_nightly_validation_cron.sh" --print
  exit 0
fi

if [ "$#" -eq 0 ]; then
  print_usage
fi

if [ "$1" = "--print" ]; then
  echo "# Add this line to your crontab to refresh FEC RAG weekly and keep logs"
  echo "$CRON_LINE"
  exit 0
fi

if [ "$1" = "--print-nightly" ]; then
  # Delegate to nightly installer in print mode (use bash to avoid relying on exec bit)
  bash "$ROOT_DIR/scripts/install_nightly_validation_cron.sh" --print
  exit 0
fi

if [ "$1" = "--install-nightly" ]; then
  bash "$ROOT_DIR/scripts/install_nightly_validation_cron.sh"
  exit 0
fi

if [ "$1" = "--install-all" ]; then
  # Install the refresh cron, then the nightly cron
  if crontab -l 2>/dev/null | grep -F -q "$CRON_CMD"; then
    echo "Refresh cron entry already present; skipping refresh install."
  else
    (crontab -l 2>/dev/null || true; echo "# FEC RAG refresh (added by BrandonBot)") | { cat; echo "$CRON_LINE"; } | crontab -
    echo "Installed cron entry for FEC RAG refresh."
  fi

  # Delegate nightly installation to its script (idempotent). Use bash to avoid exec bit issues in tests.
  bash "$ROOT_DIR/scripts/install_nightly_validation_cron.sh"
  exit 0
fi

if [ "$1" = "--install" ]; then
  # Backwards-compatible: install only the refresh cron
  if crontab -l 2>/dev/null | grep -F -q "$CRON_CMD"; then
    echo "Cron entry already present; no changes made."
    exit 0
  fi

  # Append new cron entry (preserve existing crontab)
  (crontab -l 2>/dev/null || true; echo "# FEC RAG refresh (added by BrandonBot)") | { cat; echo "$CRON_LINE"; } | crontab -
  echo "Installed cron entry for FEC RAG refresh."
  exit 0
fi

print_usage
