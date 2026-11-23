import os
import logging
from datetime import datetime

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Firestore initialisation
# ---------------------------------------------------------------------------

try:
    db_firestore = firestore.Client()
    logger.info("Firestore client initialized.")
except Exception as e:
    logger.error(f"Failed to initialize Firestore client: {e}")
    db_firestore = None

DATA_SOURCE = os.environ.get("DATA_SOURCE", "firestore").lower()

FIRESTORE_PREDICTIONS_COLLECTION = "predictions"

# ---------------------------------------------------------------------------
# Firestore document → app dict
# ---------------------------------------------------------------------------


def _firestore_doc_to_game_dict(doc_snapshot):
    """Convert a Firestore document snapshot to a game dictionary."""
    if not doc_snapshot.exists:
        return None

    game = doc_snapshot.to_dict()
    game["doc_id"] = doc_snapshot.id

    # --- Resolve unique game ID ---
    if "game_id" in game and game["game_id"]:
        game["id"] = str(game["game_id"])
    elif "prediction_source_id" in game and game["prediction_source_id"]:
        game["id"] = str(game["prediction_source_id"])
    elif "result_id" in game and game["result_id"]:
        game["id"] = str(game["result_id"])
    else:
        parts = doc_snapshot.id.split("_")
        if len(parts) > 1:
            game["id"] = str(parts[0])
        else:
            game["id"] = str(doc_snapshot.id)
            logger.warning(
                f"Could not determine game_id for doc {doc_snapshot.id}, "
                "using full doc_id"
            )

    # --- Timestamps → naive datetime ---
    for key in ["prediction_timestamp", "timestamp", "result_updated_at"]:
        if key in game and game[key] is not None:
            if isinstance(game[key], datetime):
                if game[key].tzinfo:
                    game[key] = game[key].replace(tzinfo=None)
            elif isinstance(game[key], str):
                try:
                    game[key] = datetime.fromisoformat(
                        game[key].replace("Z", "+00:00")
                    )
                    if game[key].tzinfo:
                        game[key] = game[key].replace(tzinfo=None)
                except (TypeError, ValueError) as exc:
                    logger.error(
                        f"Error converting '{key}' for doc {doc_snapshot.id}: {exc}"
                    )
                    game[key] = None

    # --- match_time (stored as "YYYY-MM-DD HH:MM" string) → datetime ---
    if "match_time" in game and isinstance(game["match_time"], str):
        try:
            game["match_time"] = datetime.strptime(game["match_time"], "%Y-%m-%d %H:%M")
        except ValueError as exc:
            logger.error(
                f"Unparsable match_time for doc {doc_snapshot.id}: {exc}. Skipping game."
            )
            return None
    elif "match_time" not in game or game["match_time"] is None:
        logger.error(f"Missing match_time for doc {doc_snapshot.id}. Skipping game.")
        return None

    # --- Numeric fields ---
    for nf in [
        "away_odds", "away_win_ev", "away_win_prob",
        "draw_ev", "draw_odds", "draw_prob",
        "expected_value", "home_odds", "home_win_ev", "home_win_prob",
        "confidence", "home_score", "away_score",
        "final_home_score", "final_away_score",
    ]:
        if nf in game and game[nf] is not None:
            try:
                game[nf] = float(game[nf])
            except (ValueError, TypeError):
                logger.warning(
                    f"Cannot convert {nf} to float for doc {doc_snapshot.id}"
                )
                game[nf] = 0.0

    # --- Boolean fields ---
    for bf in [
        "away_win_is_profitable", "draw_is_profitable",
        "home_win_is_profitable", "is_profitable", "prediction_result",
    ]:
        if bf in game and game[bf] is not None:
            game[bf] = bool(game[bf])

    # --- Odds dict ---
    if "odds" not in game or not isinstance(game["odds"], dict):
        game["odds"] = {"home": None, "draw": None, "away": None}
    else:
        for k in ["home", "draw", "away"]:
            if k not in game["odds"]:
                game["odds"][k] = None
            elif game["odds"][k] is not None:
                game["odds"][k] = float(game["odds"][k])

    # --- Derived fields (safeguards for missing data) ---
    if game.get("prediction") is None:
        hwev = game.get("home_win_ev", 0.0) or 0.0
        dev = game.get("draw_ev", 0.0) or 0.0
        awev = game.get("away_win_ev", 0.0) or 0.0
        if hwev >= dev and hwev >= awev:
            game["prediction"] = "Home Win"
        elif dev >= hwev and dev >= awev:
            game["prediction"] = "Draw"
        else:
            game["prediction"] = "Away Win"

    if game.get("confidence") is None:
        pred = game.get("prediction")
        if pred == "Home Win":
            game["confidence"] = float(game.get("home_win_prob", 0.0) or 0.0)
        elif pred == "Draw":
            game["confidence"] = float(game.get("draw_prob", 0.0) or 0.0)
        elif pred == "Away Win":
            game["confidence"] = float(game.get("away_win_prob", 0.0) or 0.0)
        else:
            game["confidence"] = 0.0

    if game.get("expected_value") is None:
        game["expected_value"] = max(
            float(game.get("home_win_ev", 0.0) or 0.0),
            float(game.get("draw_ev", 0.0) or 0.0),
            float(game.get("away_win_ev", 0.0) or 0.0),
        )

    return game


# ---------------------------------------------------------------------------
# Firestore fetch helpers. Read errors propagate: the cache then keeps its previous
# snapshot instead of replacing it with empty lists.
# ---------------------------------------------------------------------------


def fetch_pick_docs(collection, known):
    """Snapshots of a picks collection keyed by doc id. `known` = the previous result (or None).
    A settled pick never changes again, so after the first full read only pending picks and picks settled
    since the newest known settlement are read; this keeps Firestore reads flat as history grows."""
    col = db_firestore.collection(collection)
    if not known:
        return {d.id: d for d in col.stream()}
    settled = {k: d for k, d in known.items() if d.get("status") in ("completed", "void")}
    marks = [d.get("result_updated_at") for d in settled.values() if d.get("result_updated_at")]
    pending = col.where(filter=FieldFilter("status", "==", "pending")).stream()
    if marks:
        # ">=": a run settles its picks one by one with the same result_updated_at, so a read in the middle of a
        # run would otherwise never see the rest of that run's settlements
        new = col.where(filter=FieldFilter("result_updated_at", ">=", max(marks))).stream()
    else:
        new = col.where(filter=FieldFilter("status", "==", "completed")).stream()
    return {**settled, **{d.id: d for d in pending}, **{d.id: d for d in new}}


def get_profitable_games_from_firestore(docs):
    """Profitable predictions from snapshots (see fetch_pick_docs), sorted for display."""
    games = [_firestore_doc_to_game_dict(d) for d in docs if d.exists]
    games = [g for g in games if g is not None]

    if not games:
        return []

    now = datetime.now()

    games.sort(
        key=lambda x: (
            x["match_time"] <= now,
            -x["match_time"].timestamp() if x.get("match_time") else 0,
            -(x["expected_value"] or 0.0),
        )
    )

    logger.info(f"Loaded {len(games)} profitable games from Firestore.")
    return games


# Fields the games list needs; the large per-run `history` array is left out and read per game (get_prediction).
PREDICTION_FIELDS = [
    "game_id", "result_id", "match_time", "league", "english_league", "home_team", "away_team", "english_home_team",
    "english_away_team", "odds", "home_win_prob", "draw_prob", "away_win_prob", "home_win_ev", "draw_ev", "away_win_ev",
    "flagged_sides", "prediction_timestamp", "prediction", "expected_value", "confidence", "is_profitable", "model_name",
]


def get_all_predictions_from_firestore():
    """Every open game's latest prediction, without its history."""
    docs = db_firestore.collection(FIRESTORE_PREDICTIONS_COLLECTION).select(PREDICTION_FIELDS).stream()
    return [g for g in (_firestore_doc_to_game_dict(d) for d in docs) if g is not None]


def get_prediction(game_id):
    """One game's prediction including its per-run history, or None."""
    return _firestore_doc_to_game_dict(db_firestore.collection(FIRESTORE_PREDICTIONS_COLLECTION).document(game_id).get())


# ---------------------------------------------------------------------------
# Picks collections (written by lambdas/predict_winner_model)
# ---------------------------------------------------------------------------



def get_team_names():
    """Hebrew team / league name -> English (one document the predictor keeps up to date)."""
    return db_firestore.collection("meta").document("team_names").get().to_dict() or {}


def get_sweep():
    """sweep_v1 (5-minute Lambda): every position (one per paper fill; a few a week) and the last run's status."""
    positions = []
    for d in db_firestore.collection("sweep_positions").stream():
        p = d.to_dict()
        p["doc_id"] = d.id
        positions.append(p)
    status = db_firestore.collection("meta").document("sweep_status").get().to_dict() or {}
    return dict(positions=positions, status=status)


def get_slack_picks_from_firestore(docs):
    """Picks (slack_picks, oddsflow_picks, paper_bets snapshots) as plain dicts (match_time / flagged_at parsed)."""
    picks = []
    for d in docs:
        p = d.to_dict()
        p["doc_id"] = d.id
        try:
            p["match_time"] = datetime.strptime(p["match_time"], "%Y-%m-%d %H:%M")
            p["flagged_at"] = datetime.strptime(p["flagged_at"], "%Y-%m-%d %H:%M")
        except (KeyError, TypeError, ValueError):
            continue
        picks.append(p)
    return picks
