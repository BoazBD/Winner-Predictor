#!/bin/zsh
# Re-run the live rewards/book snapshot every 20 minutes (N times) to see how competing depth and reward configs evolve.
cd "$(dirname "$0")"
for i in $(seq 1 ${1:-30}); do
  /Users/boazbardavid/Desktop/Winner/winner_model/.venv/bin/python s01_rewards_live.py >> logs/s01_loop.log 2>&1
  sleep 1200
done
