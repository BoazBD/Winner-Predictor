#!/bin/bash
# Copy the pipeline + trained model from winner_model/ into this lambda's build context.
set -e
cd "$(dirname "$0")"
W=../../winner_model
cp $W/lib.py $W/build_segments.py $W/build_cand2.py $W/model.py .
cp ../../models/odds_history.py .   # the shared cleaning step (every model imports it)
cp $W/data/model_full.pkl model_full.pkl
rm -rf winner_slack_model && mkdir winner_slack_model && cp ../../models/winner_slack_model/{slack_api.py,fable_core.py} winner_slack_model/ && cp -r ../../models/winner_slack_model/model winner_slack_model/
rm -rf oddsflow && mkdir oddsflow && cp ../../models/oddsflow/{predict_api.py,features.py,table.py,model_lib.py} oddsflow/ && cp -r ../../models/oddsflow/model oddsflow/
cp ../predict_model_firestore/team_translations.py ../predict_model_firestore/service-account.json .
$W/.venv/bin/python -c "
import duckdb, json
l=[r[0] for r in duckdb.sql(\"select distinct league from '../../processed_winner_081126.parquet'\").fetchall()]
json.dump(sorted(l), open('big_leagues.json','w'), ensure_ascii=False)
print(len(l),'big leagues')"
