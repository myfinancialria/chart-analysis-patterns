#!/bin/bash
# Daily EOD pattern scan for NIFTY 500. Run after market close (~16:00 IST).
set -e
cd "$(dirname "$0")"
PY="../.venv/bin/python"
"$PY" scan.py --slack "$@"
