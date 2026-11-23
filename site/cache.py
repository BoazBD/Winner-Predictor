"""In-memory snapshot of the Firestore data the site shows.

The snapshot is loaded when the app starts (app.py), so the first request after a cold start waits for real data
instead of getting the loading page. Cloud Run only gives the container CPU while it serves a request, so there is no
background refresh: a request after the snapshot is STALE_AFTER_MINUTES old re-reads Firestore (picks incrementally,
see db.fetch_pick_docs). A Cloud Scheduler job calls /health every 10 minutes, which keeps the instance warm and does
those refreshes, so visitors rarely wait.
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

STALE_AFTER_MINUTES = 30
PICK_COLLECTIONS = ("profitable_predictions", "slack_picks", "oddsflow_picks", "paper_bets")


class DataStore:
    def __init__(self):
        self._data: Dict = {}
        self._raw: Dict = {}            # collection -> {doc_id: snapshot}, for incremental pick reads
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._last_attempt: Optional[datetime] = None

    def refresh(self):
        if not self._lock.acquire(blocking=False):
            return                      # another request is already refreshing; keep serving the old snapshot
        self._last_attempt = datetime.now()
        try:
            from db import (fetch_pick_docs, get_all_predictions_from_firestore, get_profitable_games_from_firestore,
                            get_slack_picks_from_firestore, get_sweep, get_team_names)
            with ThreadPoolExecutor(len(PICK_COLLECTIONS) + 3) as ex:     # the reads are independent: run them together
                picks = {c: ex.submit(fetch_pick_docs, c, self._raw.get(c)) for c in PICK_COLLECTIONS}
                predictions, names = ex.submit(get_all_predictions_from_firestore), ex.submit(get_team_names)
                sweep = ex.submit(get_sweep)
                raw = {c: f.result() for c, f in picks.items()}
                predictions, names = predictions.result(), names.result()
                try:                    # the sweep tab is separate: its failure must not block the rest of the site
                    sweep = sweep.result()
                except Exception:
                    logger.exception("sweep read failed")
                    sweep = self._data.get("sweep", dict(positions=[], status={}))
            self._data = {
                "profitable_games": get_profitable_games_from_firestore(raw["profitable_predictions"].values()),
                "all_predictions": predictions,
                "slack_picks": get_slack_picks_from_firestore(raw["slack_picks"].values()),
                "oddsflow_picks": get_slack_picks_from_firestore(raw["oddsflow_picks"].values()),
                "paper_bets": get_slack_picks_from_firestore(raw["paper_bets"].values()),
                "team_names": names,
                "sweep": sweep,
                "refreshed_at": datetime.now(),
            }
            self._raw = raw
            self._ready.set()
            logger.info("Data refresh done: %d predictions, %d paper bets",
                        len(self._data["all_predictions"]), len(self._data["paper_bets"]))
        except Exception:
            # keep serving the previous snapshot; the next request after STALE_AFTER_MINUTES retries
            logger.exception("Data refresh failed")
        finally:
            self._lock.release()

    def refresh_if_stale(self):
        ts = self._last_attempt
        if self._ready.is_set() and (ts is None or (datetime.now() - ts).total_seconds() > STALE_AFTER_MINUTES * 60):
            self.refresh()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def get(self, key, default=None):
        return self._data.get(key, default)

    def info(self) -> Dict:
        ts = self._data.get("refreshed_at")
        return {"ready": self.ready, "last_refresh": ts.isoformat() if ts else None,
                **{k: len(v) for k, v in self._data.items() if isinstance(v, (list, dict))}}


_store = DataStore()


def load():
    """Load synchronously (at startup)."""
    _store.refresh()


def get_cache() -> DataStore:
    """The snapshot. Until the first load succeeds, a call starts it in the background (at most once a minute)
    and pages show a loading screen."""
    if _store.ready:
        _store.refresh_if_stale()
    elif _store._last_attempt is None or (datetime.now() - _store._last_attempt).total_seconds() > 60:
        threading.Thread(target=_store.refresh, daemon=True, name="load").start()
    return _store
