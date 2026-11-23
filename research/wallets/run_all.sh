#!/bin/bash
# Re-runs the whole analysis after s01 (download). Logs to logs_<step>.txt
set -e
P=/Users/boazbardavid/Desktop/Winner/winner_model/.venv/bin/python
cd "$(dirname "$0")"
for s in ${@:-s02_build s03_marks s04_skill s05_bets s05b_ml s06_sim s07_rules}; do
  echo "== $s $(date)"; $P $s.py > logs_$s.txt 2>&1; tail -3 logs_$s.txt
done
