"""Every 3 hours: Athena (api_odds) -> winner_model pipeline -> Firestore.

Firestore layout (same schema the site already reads):
  predictions/{unique_id}                   every open soccer 1X2 game: latest score at top level + `history` array with
                                            one entry per run (so the site can show all predictions of a game); pruned at kickoff
  profitable_predictions/{unique_id}_{side} model picks: written ONCE, the first time EV crosses THRESHOLD; later settled
  meta/winner_model_state                   open games (to prune `predictions`) + de-dup store of the picks
  oddsflow_picks/{unique_id}_{outcome}      oddsflow model picks (models/oddsflow), same schema as slack_picks;
                                            meta/oddsflow_model_state = de-dup store.
  slack_picks/{unique_id}_{outcome}         "rounding slack" model picks (models/winner_slack_model): written ONCE at the
                                            first flagged snapshot, later settled. meta/slack_model_state = de-dup store.
  paper_bets/{model}_{unique_id}_{outcome}  Polymarket PAPER trades (no real money): every model's probabilities are compared
                                            with the live Polymarket ask (fee included); first time EV > POLY_MIN_EV the
                                            outcome is "bought" (1 unit) and later settled. meta/paper_state = de-dup store.
Every pick / paper bet is written with create(), so an existing one is never overwritten: the de-dup stores only
save the failed attempts (they are pruned when a market leaves the scraper run, and markets do come back).
Env: THRESHOLD (default -0.03), DRY_RUN=1 (no Firestore writes), HISTORY_DAYS (21), POLY_MIN_EV (0.03)
"""
import os, re, sys, json, pickle, logging, shutil, traceback
import numpy as np, pandas as pd
import boto3, awswrangler as wr

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from team_translations import TEAM_TRANSLATIONS as TR
logging.basicConfig(level=logging.INFO); log = logging.getLogger("winner_model"); log.setLevel(logging.INFO)

REGION = os.environ.get("CUSTOM_AWS_REGION", "il-central-1")
DB = os.environ.get("ATHENA_DATABASE", "winner-db")
ATHENA_OUT = os.environ.get("ATHENA_OUTPUT", "s3://winner-athena-output/winner_model/")
THR = float(os.environ.get("THRESHOLD", "-0.03"))
HISTORY_DAYS = int(os.environ.get("HISTORY_DAYS", "21"))
DRY = os.environ.get("DRY_RUN", "0") == "1"
MODEL_NAME = "value_v1"
TMP = "/tmp/winner_live"
PRED_COL, PICK_COL, STATE_DOC = "predictions", "profitable_predictions", "winner_model_state"
SLACK_COL, SLACK_STATE = "slack_picks", "slack_model_state"
ODF_COL, ODF_STATE = "oddsflow_picks", "oddsflow_model_state"
PAPER_COL, PAPER_STATE = "paper_bets", "paper_state"
POLY_MIN_EV = float(os.environ.get("POLY_MIN_EV", "0.03"))
# move_v1: buy when Winner's probability for an outcome rose by >= MOVE_MIN over the last MOVE_HOURS while
# Polymarket's price moved by <= MOVE_FOLLOW_MAX (it has not followed yet)
MOVE_HOURS = float(os.environ.get("MOVE_HOURS", "6"))
MOVE_MIN = float(os.environ.get("MOVE_MIN", "0.03"))
MOVE_FOLLOW_MAX = float(os.environ.get("MOVE_FOLLOW_MAX", "0.01"))
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "il-central-1")
BEDROCK_MODEL = os.environ.get("BEDROCK_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0")
PROBS = {}   # model name -> DataFrame(unique_id, outcome, p_model, winner_odds) for the latest snapshot (filled by runners)
OUTCOME = {0: "Home Win", 1: "Draw", 2: "Away Win"}
KEYS = {0: "home_win", 1: "draw", 2: "away_win"}
os.environ.setdefault("DUCKDB_MEM", "1500MB")
# big_leagues.json = the BIG_GAMES list the value model was trained with (its `big` feature): keep it unchanged.
MODEL_BIG = json.load(open(os.path.join(HERE, "big_leagues.json"), encoding="utf-8"))
# Only games that are also listed on Polymarket are predicted, picked and paper-traded (filled each run before the
# models run; the models still read all rows because their scraper-run timeline spans every market).
ON_POLY = set()

_cache = {}
NAMES = {}   # Hebrew team / league name -> English (meta/team_names: Polymarket's names for matched games + LLM translations)

def en(name):
    return TR.get(name) or NAMES.get(name) or name
def _model():
    if "m" not in _cache:
        import lightgbm as lgb
        M = pickle.load(open(os.path.join(HERE, "model_full.pkl"), "rb"))
        _cache["m"] = (M["level"], [lgb.Booster(model_str=s) for s in M["drift"]])
    return _cache["m"]

def _db():
    if "db" not in _cache:
        from google.cloud import firestore
        _cache["db"] = firestore.Client()
    return _cache["db"]

def _state(doc):
    """(reference, content) of a meta/{doc} bookkeeping document."""
    ref = _db().collection("meta").document(doc)
    return ref, ref.get().to_dict() or {}

def _create(col, pid, doc):
    """Write a pick unless it already exists: a pick keeps the odds of its first trigger."""
    from google.api_core.exceptions import Conflict
    try:
        _db().collection(col).document(pid).create(doc)
        return True
    except Conflict:
        return False

def athena(sql):
    return wr.athena.read_sql_query(sql=sql, database=DB, s3_output=ATHENA_OUT, ctas_approach=False,
                                    boto3_session=boto3.Session(region_name=REGION))

def fetch_live():
    recent = (pd.Timestamp.now("UTC") - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    latest = athena(f"""SELECT max(run_time) AS mx FROM "{DB}"."api_odds" WHERE date_parsed >= '{recent}'""").mx.iloc[0]
    if latest is None or pd.isna(latest):
        raise RuntimeError("no recent scraper run found in api_odds")
    latest_ts = pd.Timestamp(latest)
    since = (latest_ts - pd.Timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    df = athena(f"""
        SELECT event_date, time, league, event, option1, ratio1, option2, ratio2, option3, ratio3,
               run_time, result_id, unique_id, date_parsed, type
        FROM "{DB}"."api_odds"
        WHERE date_parsed >= '{since}' AND type IN ('Soccer', 'Tennis')
          AND unique_id IN (SELECT unique_id FROM "{DB}"."api_odds"
                            WHERE date_parsed >= '{recent}' AND run_time = '{latest}')""")
    log.info("fetched %d rows, %d markets, latest run %s", len(df), df.unique_id.nunique(), latest)
    return df, latest_ts

def score(df):
    import lib, build_segments, build_cand2, model
    level, boosters = _model()
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP, exist_ok=True)
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    df["run_time"] = pd.to_datetime(df["run_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in ["ratio1", "ratio2", "ratio3", "time", "league", "event", "option1", "option2", "option3", "result_id", "unique_id", "type", "date_parsed"]:
        df[c] = df[c].astype("string")
    for c in ["scorea", "scoreb"]: df[c] = pd.Series([None] * len(df), dtype="string")
    for c in ["final_scorea", "final_scoreb"]: df[c] = np.nan
    for c in ["bet1_won", "bet2_won", "tie_won"]: df[c] = pd.Series([None] * len(df), dtype="boolean")
    src = os.path.join(TMP, "live.parquet"); df.to_parquet(src, index=False)
    build_segments.build(src, TMP, BIG=MODEL_BIG, quiet=True)
    lib.D = TMP
    c = build_cand2.build(out_path=None, add_now=True).reset_index(drop=True)
    last_i = int(lib.load_runs().run_idx.max())
    c = model.prep(c)
    now = c[(c.i == last_i) & (c.hrs_to_ko > 0) & (c.mkt == "1x2")].copy()
    if now.empty: return now, lib.load_markets().set_index("unique_id")
    now["P_now"] = model.pred_level(level, now).values
    now["ev_now"] = now.P_now * now.odds - 1
    now["drift_pred"] = model.pred_drift(boosters, now)
    now["ev"] = now.ev_now + now.drift_pred
    now["p_model"] = (1 + now.ev) / now.odds
    return now, lib.load_markets().set_index("unique_id")

def build_docs(now, mk, run_ts):
    docs = {}
    for uid, g in now.groupby("unique_id"):
        if set(g.side) != {0, 1, 2}: continue
        g = g.set_index("side"); m = mk.loc[uid]
        if uid not in ON_POLY: continue
        ko = pd.Timestamp(m.kickoff)
        d = dict(game_id=uid, result_id=m.result_id, home_team=m.option1, away_team=m.option3, league=m.league,
                 english_home_team=en(m.option1), english_away_team=en(m.option3),
                 english_league=en(m.league),
                 event_date=ko.strftime("%Y-%m-%d"), game_time=ko.strftime("%H:%M"), match_time=ko.strftime("%Y-%m-%d %H:%M"),
                 prediction_timestamp=run_ts.isoformat(), timestamp=run_ts.isoformat(),
                 model_timestamp=run_ts.strftime("%b %d, %Y at %H:%M (Israel Time)"),
                 model_name=MODEL_NAME, model_type="value", status="pending", threshold=THR,
                 payout_tier={0: 0.90, 1: 0.85, 2: 0.80}[int(g.tier.iloc[0])],
                 hours_to_kickoff=round(float(g.hrs_to_ko.iloc[0]), 2),
                 odds=dict(home=round(float(g.odds[0]), 2), draw=round(float(g.odds[1]), 2), away=round(float(g.odds[2]), 2)))
        for s in (0, 1, 2):
            k = KEYS[s]; ev = float(g.ev[s])
            d[f"{k}_prob"] = round(float(g.p_model[s]), 4); d[f"{k}_ev"] = round(ev, 4)
            d[f"{k}_ev_level"] = round(float(g.ev_now[s]), 4); d[f"{k}_drift"] = round(float(g.drift_pred[s]), 4)
            d[f"{k}_is_profitable"] = bool(ev > THR)
        best = int(g.ev.idxmax())
        d.update(prediction=OUTCOME[best], expected_value=round(float(g.ev[best]), 4), confidence=d[f"{KEYS[best]}_prob"],
                 is_profitable=bool(g.ev[best] > THR), flagged_sides=[int(s) for s in (0, 1, 2) if g.ev[s] > THR])
        docs[uid] = d
    return docs

def write_firestore(docs, run_ts):
    from google.cloud import firestore
    db = _db(); stats = dict(written=0, deleted=0, new_picks=0)
    st_ref, state = _state(STATE_DOC)
    picked = set(state.get("picked", []))
    batch = db.batch(); n = 0
    def add(op, ref, *a, **k):
        nonlocal batch, n
        getattr(batch, op)(ref, *a, **k); n += 1
        if n >= 400:
            batch.commit(); batch = db.batch(); n = 0
    for uid, d in docs.items():
        entry = dict(ts=run_ts.isoformat(), odds=d["odds"], flagged_sides=d["flagged_sides"],
                     **{f"{k}_{f}": d[f"{k}_{f}"] for k in KEYS.values() for f in ("prob", "ev")})
        add("set", db.collection(PRED_COL).document(uid), {**d, "history": firestore.ArrayUnion([entry])}, merge=True)
        stats["written"] += 1
        for side in d["flagged_sides"]:
            pid = f"{uid}_{side}"
            if pid in picked: continue
            k = KEYS[side]
            p = dict(d); p.update(prediction=OUTCOME[side], expected_value=d[f"{k}_ev"], confidence=d[f"{k}_prob"], is_profitable=True,
                          pick_side=side, pick_odds=d["odds"][["home", "draw", "away"][side]], picked_at=run_ts.isoformat())
            stats["new_picks"] += _create(PICK_COL, pid, p); picked.add(pid)
    for uid in set(state.get("games", [])) - set(docs):          # game no longer open -> drop from 'predictions'
        add("delete", db.collection(PRED_COL).document(uid)); stats["deleted"] += 1
    if n: batch.commit()
    picked = sorted(p for p in picked if p.rsplit("_", 1)[0] in docs)
    st_ref.set(dict(games=sorted(docs), picked=picked, last_run=run_ts.isoformat(), threshold=THR, model=MODEL_NAME, n_open=len(docs)))
    return stats

def settle(run_ts, col=PICK_COL):
    """Fill in results for picks whose game has finished."""
    db = _db()
    pend = [(d.id, d.to_dict()) for d in db.collection(col).where("status", "==", "pending").stream()]
    cutoff = (run_ts - pd.Timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
    due = [(i, p) for i, p in pend if p.get("match_time", "9999") < cutoff and p.get("result_id")]
    if not due: return 0
    void_before = (run_ts - pd.Timedelta(hours=72)).strftime("%Y-%m-%d %H:%M")   # postponed / re-listed games
    ids = sorted({p["result_id"] for _, p in due}); since = min(p["event_date"] for _, p in due)
    inl = ", ".join("'" + str(i).replace("'", "") + "'" for i in ids)
    res = athena(f"""SELECT id, scorea, scoreb FROM "{DB}"."results" WHERE date >= '{since}' AND id IN ({inl})""")
    res = res.dropna(subset=["scorea", "scoreb"]).drop_duplicates("id").set_index("id"); done = 0
    for doc_id, p in due:
        if p["result_id"] not in res.index:
            if p["match_time"] < void_before:
                db.collection(col).document(doc_id).update(dict(status="void", profit=0.0, result_updated_at=run_ts.isoformat()))
                done += 1
            continue
        a, b = int(float(res.at[p["result_id"], "scorea"])), int(float(res.at[p["result_id"], "scoreb"]))
        actual = "Home Win" if a > b else "Draw" if a == b else "Away Win"
        won = (actual == p["prediction"]) != bool(p.get("bet_no"))      # bet_no: the bet is AGAINST the outcome
        db.collection(col).document(doc_id).update(dict(
            status="completed", actual_result=actual, prediction_result=won, home_score=a, away_score=b,
            final_home_score=a, final_away_score=b, profit=round((p.get("pick_odds", 0) if won else 0) - 1, 4),
            result_updated_at=run_ts.isoformat(), **(closing_line(p) if col == PAPER_COL else {})))
        done += 1
    return done

def _utc(local):
    """Naive Israel-local time (Winner) -> aware UTC."""
    return pd.Timestamp(local).tz_localize("Asia/Jerusalem").tz_convert("UTC")

def closing_line(p):
    """Polymarket price of a paper bet's outcome at kickoff, and its closing-line value (CLV): what the share was worth
    at kickoff relative to what it cost (fee included). Average CLV > 0 over many bets = the bets beat the market;
    it needs far fewer bets than profit to tell."""
    import polymarket as P
    try:
        tok = p.get("poly_token") or P.token_for_market(p["poly_market"])
        close = P.price_at(tok, _utc(p["match_time"]) + pd.Timedelta(minutes=1)) if tok else None
    except Exception as e:
        log.warning("closing price for %s failed: %s", p.get("poly_market"), e)
        return {}
    if close is None: return {}
    if p.get("bet_no"): close = 1 - close                      # the bet holds the "No" share
    return dict(poly_close=round(close, 4), clv=round(close * p["poly_odds"] - 1, 4))

# ---------------------------------------------------------------------------------------------
# Slack model (models/winner_slack_model): favourite, odds<=1.45, 90%-payout tier, rounding slack>=1%
# ---------------------------------------------------------------------------------------------
def slack_new_picks(df, run_ts, last_run, picked):
    """Score every snapshot of the open markets and return each (market, outcome)'s FIRST flagged snapshot
    that is new since the previous lambda run (first run: only the current snapshot counts)."""
    if "slack" not in _cache:
        sys.path.insert(0, os.path.join(HERE, "winner_slack_model"))
        from slack_api import SlackModel
        _cache["slack"] = SlackModel()
    out = _cache["slack"].score(df)
    if out.empty: return [], 0
    latest = out.run_time.max()
    cur = out[out.run_time == latest]
    PROBS["slack_v1"] = pd.DataFrame(dict(unique_id=cur.unique_id.values, outcome=cur.outcome.astype(int).values,
                                          p_model=cur.p_model.values, winner_odds=cur.odds.values))
    since = pd.Timestamp(last_run) if last_run else latest - pd.Timedelta(seconds=1)
    f = out[out.flag & (out.run_time > since) & (out.kick > latest) & out.unique_id.isin(ON_POLY)].sort_values("run_time")
    f = f.drop_duplicates(["unique_id", "outcome"])
    picks = []
    for r in f.itertuples():
        pid = f"{r.unique_id}_{int(r.outcome)}"
        if pid in picked: continue
        ko = pd.Timestamp(r.kick)
        picks.append((pid, dict(
            game_id=r.unique_id, result_id=r.result_id, league=r.league, english_league=en(r.league),
            home_team=r.home, away_team=r.away, english_home_team=en(r.home), english_away_team=en(r.away),
            event_date=ko.strftime("%Y-%m-%d"), match_time=ko.strftime("%Y-%m-%d %H:%M"),
            outcome=int(r.outcome), prediction=OUTCOME[int(r.outcome)], pick=r.pick, english_pick="Draw" if int(r.outcome) == 1 else (en(r.pick)),
            pick_odds=round(float(r.odds), 2), p_model=round(float(r.p_model), 4), p_mkt=round(float(r.p_mkt), 4),
            ev=round(float(r.ev), 4), rr=round(float(r.rr), 4), M=round(float(r.M), 3),
            flagged_at=pd.Timestamp(r.run_time).strftime("%Y-%m-%d %H:%M"), recorded_at=run_ts.isoformat(),
            model_name="slack_v1", status="pending")))
    return picks, int(cur.flag.sum())

def run_picks(name, col, state_doc, new_picks, df, run_ts, dry):
    """Shared runner of the picks-only models: record the new first-trigger picks, then settle finished ones."""
    if dry:
        picks, n_now = new_picks(df, run_ts, None, set())
        return {f"{name}_flagged_now": n_now, f"{name}_new_picks": [
            (p["english_home_team"], p["english_away_team"], p["english_pick"], p["pick_odds"], p["ev"]) for _, p in picks]}
    st_ref, st = _state(state_doc)
    picked = set(st.get("picked", []))
    picks, n_now = new_picks(df, run_ts, st.get("last_run"), picked)
    new = sum(_create(col, pid, p) for pid, p in picks)
    open_ids = set(df.unique_id.astype(str))
    picked = sorted(p for p in picked | {pid for pid, _ in picks} if p.rsplit("_", 1)[0] in open_ids)
    st_ref.set(dict(last_run=run_ts.isoformat(), picked=picked))
    return {f"{name}_flagged_now": n_now, f"{name}_new_picks": new, f"{name}_settled": settle(run_ts, col)}

def run_slack(df, run_ts, dry):
    return run_picks("slack", SLACK_COL, SLACK_STATE, slack_new_picks, df, run_ts, dry)

# ---------------------------------------------------------------------------------------------
# oddsflow model (models/oddsflow): drift LightGBM (numpy trees) + favourite-longshot calibration, bet when EV > 0
# ---------------------------------------------------------------------------------------------
def oddsflow_new_picks(df, run_ts, last_run, picked):
    """First-trigger picks: rows where the rule fires for the first time in the history (flag & ~flag_before),
    restricted to snapshots after the previous lambda run (first run: only the current snapshot)."""
    if "oddsflow" not in _cache:
        sys.path.insert(0, os.path.join(HERE, "oddsflow"))
        from predict_api import Model
        _cache["oddsflow"] = Model()
    raw = df.astype(object).where(df.notna(), None)    # Athena gives nullable dtypes; the model expects plain objects
    out = _cache["oddsflow"].score(raw, latest=False)
    if out.empty: return [], 0
    out["run_time"] = pd.to_datetime(out.run_time); out["kick"] = pd.to_datetime(out.kick)
    latest = out.run_time.max()
    cur = out[out.run_time == latest]
    PROBS["oddsflow_v1"] = pd.DataFrame(dict(unique_id=cur.unique_id.values, outcome=cur.outcome.astype(int).values,
                                             p_model=cur.p_model.values, winner_odds=cur.odds.values))
    since = pd.Timestamp(last_run).tz_localize(None) if last_run else latest - pd.Timedelta(seconds=1)
    f = out[out.flag & ~out.flag_before & (out.run_time > since) & (out.kick > latest)].sort_values("run_time")
    f = f.drop_duplicates(["unique_id", "outcome"])
    meta = df.drop_duplicates("unique_id").set_index("unique_id")[["league", "option1", "option3"]]
    picks = []
    for r in f.itertuples():
        pid = f"{r.unique_id}_{int(r.outcome)}"
        if pid in picked or r.unique_id not in meta.index or r.unique_id not in ON_POLY: continue
        m = meta.loc[r.unique_id]; ko = pd.Timestamp(r.kick); o = int(r.outcome)
        home, away, league = m.option1, m.option3, m.league
        pick = {0: home, 1: "תיקו", 2: away}[o]
        picks.append((pid, dict(
            game_id=r.unique_id, result_id=r.result_id, league=league, english_league=en(league),
            home_team=home, away_team=away, english_home_team=en(home), english_away_team=en(away),
            event_date=ko.strftime("%Y-%m-%d"), match_time=ko.strftime("%Y-%m-%d %H:%M"),
            outcome=o, prediction=OUTCOME[o], pick=pick, english_pick="Draw" if o == 1 else (en(pick)),
            pick_odds=round(float(r.odds), 2), p_model=round(float(r.p_model), 4), p_mkt=round(float(r.p_market), 4),
            ev=round(float(r.ev), 4), drift=round(float(r.drift_hat), 4), payout_tier={0: 0.90, 1: 0.85, 2: 0.80}.get(int(r.tier)),
            flagged_at=pd.Timestamp(r.run_time).strftime("%Y-%m-%d %H:%M"), recorded_at=run_ts.isoformat(),
            model_name="oddsflow_v1", status="pending")))
    return picks, int((cur.flag & ~cur.flag_before).sum())

def run_oddsflow(df, run_ts, dry):
    return run_picks("oddsflow", ODF_COL, ODF_STATE, oddsflow_new_picks, df, run_ts, dry)

def run_value(df, run_ts, dry):
    """winner_model value model: score every open 1X2 game, pick when EV > THRESHOLD."""
    now, mk = score(df)
    if len(now):
        PROBS[MODEL_NAME] = pd.DataFrame(dict(unique_id=now.unique_id.values, outcome=now.side.astype(int).values,
                                              p_model=now.p_model.values, winner_odds=now.odds.values))
    docs = build_docs(now, mk, run_ts) if len(now) else {}
    flagged = [(d["english_home_team"], d["english_away_team"], d["prediction"], d["expected_value"]) for d in docs.values() if d["is_profitable"]]
    out = dict(games_scored=len(docs), flagged_now=len(flagged), threshold=THR)
    if dry:
        out["flagged"] = flagged[:20]
    else:
        out.update(write_firestore(docs, run_ts)); out["settled"] = settle(run_ts)
    return out

# ---------------------------------------------------------------------------------------------
# Polymarket paper trading: same model probabilities, priced against the live Polymarket order book
# ---------------------------------------------------------------------------------------------
def winner_games(df):
    """Latest-run soccer 1X2 games with kickoff in UTC (Winner times are Israel local)."""
    s = df[(df.type.astype(str) == "Soccer") & df.event.astype(str).str.contains("1X2", regex=False)]
    s = s[s.run_time.astype(str) == str(df.run_time.max())].drop_duplicates("unique_id")
    kick = pd.to_datetime(s.event_date.astype(str).str[:10] + " " + s.time.astype(str), errors="coerce")
    g = pd.DataFrame(dict(unique_id=s.unique_id.astype(str).values, result_id=s.result_id.astype(str).values,
                          home=s.option1.astype(str).values, away=s.option3.astype(str).values, league=s.league.astype(str).values,
                          kick_local=kick.values,
                          kick_utc=kick.dt.tz_localize("Asia/Jerusalem", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC").dt.tz_localize(None).values))
    return g.dropna(subset=["kick_utc"])

def team_names(games, dry):
    """Hebrew -> English team names (Firestore meta/team_names, one document). Names not seen before are translated
    by the LLM on Bedrock and saved, so each team is translated once. If Bedrock fails, matching runs on sound only."""
    import polymarket as P
    ref = _db().collection("meta").document("team_names")
    names = ref.get().to_dict() or {}
    todo = sorted({(n, lg) for col in ("home", "away") for n, lg in zip(games[col], games.league) if n not in names})
    todo = todo[:100]                   # one LLM call per run keeps the run inside the Lambda timeout; the rest next run
    if todo:
        try:
            new = P.translate(todo, BEDROCK_REGION, BEDROCK_MODEL)
            log.info("translated %d of %d new team names", len(new), len(todo))
            names.update(new)
            if new and not dry:
                ref.set(new, merge=True)
        except Exception as e:
            log.error("team name translation failed: %s", e)
    return names

def learn_names(x, dry, min_score=0.8):
    """Polymarket's English names of confidently matched games become the names shown on the site (and exact
    aliases for later matching). Leagues get Polymarket's series title."""
    new = {}
    for r in x.itertuples():            # every accepted match names its league; team names need a confident match
        home, away = (r.poly_away, r.poly_home) if r.swapped else (r.poly_home, r.poly_away)
        league = re.sub(r"\s+20\d\d(?:[/-]\d{2,4})?$", "", r.series_title or "")   # "Süper Lig 2025" -> "Süper Lig"
        pairs = ((r.home, home), (r.away, away), (r.league, league)) if r.score >= min_score else ((r.league, league),)
        for he, eng in pairs:
            if eng and NAMES.get(he) != eng and he not in TR:
                new[he] = eng
    NAMES.update(new)
    if new and not dry:
        _db().collection("meta").document("team_names").set(new, merge=True)

def poly_quotes(df, dry):
    """Winner games matched to Polymarket, in long form (one row per Winner outcome) with Polymarket prices."""
    import polymarket as P
    poly = P.fetch_games(days=8)
    if poly.empty:                      # an empty answer is an API problem; without this every prediction would be deleted
        raise RuntimeError("Polymarket returned no soccer games")
    games = winner_games(df)
    NAMES.clear(); NAMES.update(team_names(games, dry))
    m = P.match(games, poly, NAMES)
    if m.empty: return pd.DataFrame(), len(poly), len(games)
    x = m.merge(games, on="unique_id").merge(poly, on="event_id")
    learn_names(x, dry)
    rows = []
    for r in x.itertuples():
        mids = [((getattr(r, f"bid{k}") or 0) + getattr(r, f"ask{k}")) / 2 for k in range(3)]
        tot = sum(mids)
        for o in range(3):
            k = {0: 2, 1: 1, 2: 0}[o] if r.swapped else o          # Polymarket side of Winner outcome o
            ask, bid, fee = getattr(r, f"ask{k}"), getattr(r, f"bid{k}"), getattr(r, f"fee{k}")
            rows.append(dict(unique_id=r.unique_id, result_id=r.result_id, home=r.home, away=r.away, league=r.league,
                             kick_local=r.kick_local, outcome=o, event_id=r.event_id, poly_title=r.title, poly_slug=r.slug,
                             poly_market=getattr(r, f"mkt{k}"), poly_token=getattr(r, f"tok{k}"),
                             poly_cond=getattr(r, f"cond{k}"),
                             match_score=r.score, swapped=r.swapped,
                             poly_bid=bid, poly_ask=ask, poly_fee_rate=fee, poly_liq=getattr(r, f"liq{k}"),
                             poly_p_mid=mids[k] / tot if tot > 0 else np.nan,
                             poly_odds=P.effective_odds(ask, fee) if ask and ask > 0 else np.nan))
    return pd.DataFrame(rows), len(poly), len(games)

def winner_moves(df):
    """Winner's margin-free probability (1/odds normalised) of every outcome in the latest run, and its change since
    MOVE_HOURS earlier (or since the market opened, if later). Read from the de-duplicated history (odds_history):
    the price at a time is that of the last segment started by then."""
    from odds_history import soccer_1x2_history
    seg, runs = soccer_1x2_history(df)
    cur = seg[seg.i_end == len(runs) - 1].set_index("unique_id")
    then = seg[seg.t_start <= runs[-1] - pd.Timedelta(hours=MOVE_HOURS)].groupby("unique_id").tail(1).set_index("unique_id")
    then = then.combine_first(seg.groupby("unique_id").head(1).set_index("unique_id")).reindex(cur.index)
    inv_now, inv_then = 1 / cur[["r1", "r2", "r3"]].values, 1 / then[["r1", "r2", "r3"]].values
    p_now, p_then = inv_now / inv_now.sum(1, keepdims=True), inv_then / inv_then.sum(1, keepdims=True)
    return pd.concat([pd.DataFrame(dict(unique_id=cur.index, outcome=o, winner_odds=cur[f"r{o + 1}"].values,
                                        p_w_now=p_now[:, o], w_move=p_now[:, o] - p_then[:, o])) for o in range(3)])

def _tradable(x, run_ts):
    return x[x.poly_ask.between(0.03, 0.97) & ((x.poly_ask - x.poly_bid) <= 0.05) & (x.kick_local > run_ts)]

def move_candidates(q, moves, run_ts):
    """move_v1: Winner's probability rose by >= MOVE_MIN while Polymarket has not followed (moved <= MOVE_FOLLOW_MAX).
    Expected Polymarket probability = its price MOVE_HOURS ago + Winner's move; bet when that beats the cost."""
    import polymarket as P
    x = _tradable(q.merge(moves, on=["unique_id", "outcome"]), run_ts)
    x = x[x.w_move >= MOVE_MIN].copy()
    if x.empty: return x
    then = _utc(run_ts) - pd.Timedelta(hours=MOVE_HOURS)
    x["poly_then"] = [P.price_at(t, then) if t else None for t in x.poly_token]
    x = x.dropna(subset=["poly_then"])
    x["poly_move"] = (x.poly_bid + x.poly_ask) / 2 - x.poly_then
    x = x[x.poly_move <= MOVE_FOLLOW_MAX].copy()
    x["p_model"] = (x.poly_then + x.w_move).clip(upper=0.99)
    x["ev"] = x.p_model * x.poly_odds - 1
    x = x[x.ev > POLY_MIN_EV].copy(); x["model_name"] = "move_v1"
    return x

def paper_candidates(q, run_ts):
    """Rows where a model's probability beats the Polymarket price (fee included) by POLY_MIN_EV."""
    out = []
    for name, pr in PROBS.items():
        x = _tradable(q.merge(pr, on=["unique_id", "outcome"]), run_ts)
        x = x[(x.p_model - x.poly_p_mid).abs() <= 0.15]        # bigger gaps = wrong match / stale book, not an edge
        x["ev"] = x.p_model * x.poly_odds - 1
        x = x[x.ev > POLY_MIN_EV].copy(); x["model_name"] = name
        out.append(x)
    return pd.concat(out) if out else pd.DataFrame()

def run_polymarket(q, n_poly, n_games, df, run_ts, dry):
    import polymarket as P
    res = dict(poly_games=n_poly, winner_games=n_games, matched=int(q.unique_id.nunique()) if len(q) else 0, min_ev=POLY_MIN_EV)
    if q.empty: return res
    moves = winner_moves(df)
    c = pd.concat([paper_candidates(q, run_ts), move_candidates(q, moves, run_ts)], ignore_index=True)
    if len(c):   # every bet records how both prices moved before it, to compare bets by who moved first
        c = c.drop(columns=["w_move"], errors="ignore").merge(moves[["unique_id", "outcome", "w_move"]], on=["unique_id", "outcome"], how="left")
    if dry:
        res["opportunities"] = [(r.model_name, r.home, r.away, OUTCOME[r.outcome], round(r.winner_odds, 2), round(r.poly_odds, 3), round(r.ev, 3),
                                 round(r.w_move, 3))
                                for r in c.itertuples()][:40] if len(c) else []
        return res
    st_ref, st = _state(PAPER_STATE)
    picked = set(st.get("picked", [])); new = 0
    for r in c.itertuples():
        pid = f"{r.model_name}_{r.unique_id}_{r.outcome}"
        if pid in picked: continue
        ko = pd.Timestamp(r.kick_local)
        pick = {0: r.home, 1: "תיקו", 2: r.away}[r.outcome]
        poly_move = getattr(r, "poly_move", None)
        if (poly_move is None or pd.isna(poly_move)) and r.poly_token:
            try:
                then = P.price_at(r.poly_token, _utc(run_ts) - pd.Timedelta(hours=MOVE_HOURS))
                poly_move = None if then is None else (r.poly_bid + r.poly_ask) / 2 - then
            except Exception as e:
                log.warning("price history for %s failed: %s", r.poly_market, e); poly_move = None
        new += _create(PAPER_COL, pid, dict(
            model_name=r.model_name, game_id=r.unique_id, result_id=r.result_id, league=r.league,
            english_league=en(r.league), home_team=r.home, away_team=r.away,
            english_home_team=en(r.home), english_away_team=en(r.away),
            poly_title=r.poly_title, poly_slug=r.poly_slug, poly_market=r.poly_market, match_score=float(r.match_score),
            event_date=ko.strftime("%Y-%m-%d"), match_time=ko.strftime("%Y-%m-%d %H:%M"),
            outcome=int(r.outcome), prediction=OUTCOME[r.outcome], pick=pick,
            english_pick="Draw" if r.outcome == 1 else (en(pick)),
            p_model=round(float(r.p_model), 4), winner_odds=round(float(r.winner_odds), 2),
            poly_ask=round(float(r.poly_ask), 4), poly_bid=round(float(r.poly_bid), 4), poly_p_mid=round(float(r.poly_p_mid), 4),
            poly_fee_rate=float(r.poly_fee_rate), poly_odds=round(float(r.poly_odds), 4), pick_odds=round(float(r.poly_odds), 4),
            poly_liquidity=round(float(r.poly_liq), 2), ev=round(float(r.ev), 4), stake=1.0, poly_token=r.poly_token,
            winner_move_6h=None if pd.isna(r.w_move) else round(float(r.w_move), 4),
            poly_move_6h=None if poly_move is None or pd.isna(poly_move) else round(float(poly_move), 4),
            flagged_at=run_ts.strftime("%Y-%m-%d %H:%M"), recorded_at=run_ts.isoformat(), status="pending"))
        picked.add(pid)
    open_ids = set(df.unique_id.astype(str))
    st_ref.set(dict(last_run=run_ts.isoformat(), min_ev=POLY_MIN_EV, picked=sorted(p for p in picked if p.rsplit("_", 2)[1] in open_ids)))
    res.update(new_paper_bets=new, opportunities_now=len(c), paper_settled=settle(run_ts, PAPER_COL))
    return res

# ---------------------------------------------------------------------------------------------
# longshot_v1 (research/edge): people overpay for longshots on Polymarket, so sell them. For every outcome priced
# LS_LO..LS_HI within LS_HOURS of kickoff, a resting limit order sells its "Yes" share (= buys "No") at the mid rounded
# up to the cent. The order counts as filled only when a real trade later went THROUGH that price before kickoff
# (polymarket.max_yes_buy); a filled order becomes a paper bet against the outcome, unfilled orders are cancelled.
# Research (16k games, real trades): +2.7% / +3.9% / +5.3% per $ at risk in train / check / final-test periods;
# no edge above 10c; timing within 72h does not matter.
# ---------------------------------------------------------------------------------------------
LS_COL, LS_LO, LS_HI, LS_HOURS, LS_MAX_SPREAD = "longshot_orders", 0.02, 0.10, 12, 0.05
LS_TRUE = 0.65        # filled longshots sold at price a won 0.65*a of the time (train period): the model's probability

def run_longshot(q, run_ts, dry, time_left=lambda: 1e9):
    import polymarket as P
    if q.empty: return {}
    mid = (q.poly_bid + q.poly_ask) / 2
    x = q[(q.kick_local > run_ts) & (q.kick_local <= run_ts + pd.Timedelta(hours=LS_HOURS)) & mid.between(LS_LO, LS_HI)
          & ((q.poly_ask - q.poly_bid) <= LS_MAX_SPREAD) & q.poly_cond.notna()].copy()
    x["sell"] = np.ceil((x.poly_bid + x.poly_ask) / 2 * 100 - 1e-6) / 100
    if dry:
        return dict(candidates=[(r.home, r.away, OUTCOME[r.outcome], r.poly_bid, r.poly_ask, r.sell) for r in x.itertuples()])
    db = _db(); now = pd.Timestamp.now("UTC")
    open_orders = [(d.id, d.to_dict()) for d in db.collection(LS_COL).where("status", "==", "open").stream()]
    res = dict(placed=0, filled=0, cancelled=0, open=0)
    for r in x.itertuples():                                   # place (write-once: an order is never re-priced)
        ko = pd.Timestamp(r.kick_local); a = float(r.sell); odds = round(1 / (1 - a), 4); p_no = 1 - LS_TRUE * a
        pick = {0: r.home, 1: "תיקו", 2: r.away}[r.outcome]
        bet = dict(model_name="longshot_v1", bet_no=True, game_id=r.unique_id, result_id=r.result_id, league=r.league,
                   english_league=en(r.league), home_team=r.home, away_team=r.away,
                   english_home_team=en(r.home), english_away_team=en(r.away),
                   poly_title=r.poly_title, poly_slug=r.poly_slug, poly_market=r.poly_market, match_score=float(r.match_score),
                   event_date=ko.strftime("%Y-%m-%d"), match_time=ko.strftime("%Y-%m-%d %H:%M"),
                   outcome=int(r.outcome), prediction=OUTCOME[r.outcome], pick=pick,
                   english_pick="Draw" if r.outcome == 1 else en(pick), sell_price=a,
                   p_model=round(p_no, 4), poly_ask=round(1 - a, 4), poly_bid=round(float(r.poly_bid), 4),
                   poly_fee_rate=0.0, poly_odds=odds, pick_odds=odds, ev=round(p_no * odds - 1, 4), stake=1.0,
                   poly_token=r.poly_token, status="pending")
        res["placed"] += _create(LS_COL, f"{r.unique_id}_{r.outcome}", dict(
            status="open", placed_at=now.isoformat(), kick_utc=_utc(ko).isoformat(), poly_cond=r.poly_cond, sell_price=a, bet=bet))
    for oid, o in open_orders:                                 # check the orders of earlier runs against real trades
        if time_left() < 45: break
        kick = pd.Timestamp(o["kick_utc"]); a = o["sell_price"]
        try:
            best, complete = P.max_yes_buy(o["poly_cond"], o["placed_at"], min(now, kick))
        except Exception as e:
            log.warning("trades for order %s failed: %s", oid, e); continue
        ref = db.collection(LS_COL).document(oid)
        if best is not None and best > a + 1e-9:
            _create(PAPER_COL, f"longshot_v1_{oid}", dict(o["bet"], flagged_at=run_ts.strftime("%Y-%m-%d %H:%M"), recorded_at=run_ts.isoformat()))
            ref.update(dict(status="filled", fill_seen_at=now.isoformat(), max_trade=best)); res["filled"] += 1
        elif now >= kick:
            ref.update(dict(status="cancelled", max_trade=best, trades_complete=complete)); res["cancelled"] += 1
        else:
            res["open"] += 1
    return res

# ---------------------------------------------------------------------------------------------
# sweep_v1 (research/ideas/sweep_catch) runs in its own 5-minute Lambda (sweep.py). This run hands it the matched
# games kicking off in the next 12 h: Polymarket markets / tokens and Winner's margin-free probability per outcome
# (its filter: only buy when the price fell well below Winner's fair value).
# ---------------------------------------------------------------------------------------------
SWEEP_HORIZON_H = 12

def write_sweep_universe(q, df, run_ts, dry):
    if q.empty: return {"games": 0}
    moves = winner_moves(df)[["unique_id", "outcome", "p_w_now"]]
    x = q.merge(moves, on=["unique_id", "outcome"], how="left")
    x = x[(x.kick_local > run_ts) & (x.kick_local <= run_ts + pd.Timedelta(hours=SWEEP_HORIZON_H)) & x.poly_cond.notna()
          & x.poly_token.notna()]
    games = []
    for uid, g in x.groupby("unique_id"):
        if len(g) != 3: continue
        r = g.iloc[0]; ko = pd.Timestamp(r.kick_local)
        games.append(dict(
            uid=uid, result_id=r.result_id, home=r.home, away=r.away, en_home=en(r.home), en_away=en(r.away),
            league=r.league, en_league=en(r.league), poly_title=r.poly_title, poly_slug=r.poly_slug,
            match_time=ko.strftime("%Y-%m-%d %H:%M"), kick_ts=int(_utc(ko).timestamp()),
            outcomes=[dict(o=int(o.outcome), cond=o.poly_cond, token=o.poly_token,
                           wf=None if pd.isna(o.p_w_now) else round(float(o.p_w_now), 4),
                           fee_rate=float(o.poly_fee_rate) if pd.notna(o.poly_fee_rate) else 0.05)
                      for o in g.sort_values("outcome").itertuples()]))
    if not dry:
        _db().collection("meta").document("sweep_universe").set(dict(
            updated=pd.Timestamp.now("UTC").isoformat(), winner_run=str(run_ts), games=games))
    return {"games": len(games)}

# Every model the predictor runs. To add one: write run_x(df, run_ts, dry) -> dict and list it here
# (and add its name to MODELS in site/app.py so it appears in the home-page selector).
MODEL_RUNNERS = [("value_v1", run_value), ("slack_v1", run_slack), ("oddsflow_v1", run_oddsflow)]

def lambda_handler(event, context):
    try:
        df, run_ts = fetch_live()
    except Exception as e:
        log.error("fetch failed: %s\n%s", e, traceback.format_exc())
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
    dry = DRY or bool((event or {}).get("dry_run"))
    age_h = (pd.Timestamp.now("Asia/Jerusalem").tz_localize(None) - pd.Timestamp(run_ts)).total_seconds() / 3600
    if age_h > 2:                       # stale Winner odds vs live Polymarket prices would create fake edges
        return {"statusCode": 200, "body": json.dumps({"skipped": f"latest Winner run is {age_h:.1f} h old"})}
    try:                                # Polymarket first: it decides which games are in scope this run
        q, n_poly, n_games = poly_quotes(df, dry)
    except Exception as e:              # keep the site as it is rather than wiping predictions
        log.error("polymarket fetch failed: %s\n%s", e, traceback.format_exc())
        return {"statusCode": 500, "body": json.dumps({"error": f"polymarket: {e}"})}
    ON_POLY.clear(); ON_POLY.update(q.unique_id.unique() if len(q) else [])
    out = dict(run=str(run_ts), dry_run=dry, on_polymarket=len(ON_POLY), models={}); failed = 0; PROBS.clear()
    try:                                # sweep_v1 (separate 5-minute Lambda) reads its games from here
        out["sweep_universe"] = write_sweep_universe(q, df, run_ts, dry)
    except Exception as e:
        log.error("sweep universe failed: %s\n%s", e, traceback.format_exc()); out["sweep_universe"] = {"error": str(e)}
    for name, runner in MODEL_RUNNERS:
        try:
            out["models"][name] = runner(df, run_ts, dry)
        except Exception as e:          # models are independent: one failing must not stop the others
            failed += 1; log.error("model %s failed: %s\n%s", name, e, traceback.format_exc())
            out["models"][name] = {"error": str(e)}
    try:                                # after the models: they fill PROBS
        out["polymarket"] = run_polymarket(q, n_poly, n_games, df, run_ts, dry)
    except Exception as e:
        log.error("polymarket failed: %s\n%s", e, traceback.format_exc()); out["polymarket"] = {"error": str(e)}
    try:                                # longshot_v1: resting orders against longshots, independent of the models
        left = (lambda: context.get_remaining_time_in_millis() / 1000) if context is not None else (lambda: 1e9)
        out["longshot"] = run_longshot(q, run_ts, dry, left)
    except Exception as e:
        log.error("longshot failed: %s\n%s", e, traceback.format_exc()); out["longshot"] = {"error": str(e)}
    log.info("result %s", out)
    return {"statusCode": 200 if failed < len(MODEL_RUNNERS) else 500, "body": json.dumps(out, ensure_ascii=False, default=str)}

if __name__ == "__main__":
    print(lambda_handler({}, None))
