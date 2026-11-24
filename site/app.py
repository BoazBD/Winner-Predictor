"""Website for the Winner models' paper trading on Polymarket (data written by lambdas/predict_winner_model)."""
import logging
import os
import re
from datetime import datetime
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from flask import Flask, Response, redirect, render_template, request, url_for

from cache import get_cache, load

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
load()   # load before serving: a cold start then answers with real data instead of the loading page

# Model order fixes each model's chart colour (series-1..4); never reorder an existing model.
MODELS = {
    'value_v1': dict(
        label='Value model', picks='profitable_games',
        description="Corrects Winner's favourite–longshot bias and price rounding, and predicts where Winner's odds "
                    "will move before kickoff.",
        description_he="מתקן את ההטיה של Winner לטובת אנדרדוגים ואת עיגול היחסים, וחוזה לאן יזוזו היחסים עד תחילת המשחק."),
    'slack_v1': dict(
        label='Slack model', picks='slack_picks',
        description="Backs short favourites (odds ≤ 1.45) in Winner's lowest-margin leagues when the price was "
                    "rounded in the bettor's favour.",
        description_he="מהמר על פייבוריטים ביחס של עד 1.45 בליגות עם העמלה הנמוכה ב-Winner, כשהיחס עוגל לטובת המהמר."),
    'oddsflow_v1': dict(
        label='Oddsflow model', picks='oddsflow_picks',
        description="Reads how a match's odds have moved, predicts the closing price and turns it into probabilities "
                    "with a favourite–longshot correction.",
        description_he="קורא איך היחסים של המשחק זזו, חוזה את היחס הסופי והופך אותו להסתברויות עם תיקון הטיית פייבוריט-אנדרדוג."),
    'move_v1': dict(
        label='Winner-move model', picks=None,
        description="Buys on Polymarket when Winner's probability for an outcome rose by 3+ points in 6 hours and "
                    "Polymarket has not followed (moved ≤ 1 point).",
        description_he="קונה בפולימרקט כשההסתברות של Winner לתוצאה עלתה ב-3 נקודות או יותר ב-6 שעות ופולימרקט עוד לא הגיב (זז נקודה אחת לכל היותר)."),
    'longshot_v1': dict(
        label='Longshot seller', picks=None,
        description="Bets AGAINST longshots: for every outcome priced 2–10¢ within 12 hours of kickoff it rests a "
                    "limit order to buy the \"No\" share (no fee). Only orders that a real Polymarket trade went "
                    "through count as bets; the rest are cancelled at kickoff. \"Not X\" wins unless X happens.",
        description_he="מהמר נגד אנדרדוגים: לכל תוצאה שמחירה 2–10 סנט ב-12 השעות שלפני המשחק מוצבת פקודת לימיט לקניית "
                       "מניית ה-\"לא\" (ללא עמלה). רק פקודות שעסקה אמיתית בפולימרקט עברה דרכן נספרות; השאר מבוטלות "
                       "בשריקת הפתיחה. \"לא X\" זוכה אלא אם X קורה."),
}
EV_OPTIONS = [None, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15]
# Public address, for canonical links, the sitemap and social previews (set SITE_URL when a custom domain is added).
SITE_URL = os.environ.get('SITE_URL', 'https://winner-site-dypbatdkoq-uc.a.run.app').rstrip('/')
ISRAEL_TZ = ZoneInfo('Asia/Jerusalem')

# Search-result title and description of each page (the game page builds its own from the teams).
SEO = {
    'en': dict(
        dashboard=('AI soccer betting models tested live on Polymarket',
                   'Five AI models built on Winner (Israel) odds, paper-traded against Polymarket soccer prices: live '
                   'bets, profit and loss, ROI and closing-line value for every model, updated every 3 hours.'),
        games=("Today's soccer games: Winner odds and AI predictions",
               'Every open soccer game on both Winner and Polymarket, with Winner odds, AI win / draw / loss '
               'probabilities and expected value. Updated every 3 hours.'),
        sweep=('Panic catcher: buying panic sales on Polymarket soccer',
               'A live paper-trading bot that rests deep buy orders before kickoff and catches panic sales on '
               'Polymarket soccer markets: every fill, exit and P&L.'),
        about=('How the AI betting models work',
               'How the site works: Winner odds, Polymarket prices, game matching, the five models and how paper '
               'bets are placed and settled.')),
    'he': dict(
        dashboard=('מודלים של AI להימורי כדורגל מול פולימרקט',
                   'חמישה מודלים של AI שמבוססים על יחסי Winner, במסחר מדומה מול מחירי כדורגל בפולימרקט: הימורים חיים, '
                   'רווח והפסד, תשואה ו-CLV לכל מודל, מתעדכן כל 3 שעות.'),
        games=('משחקי הכדורגל של היום: יחסי Winner ותחזיות AI',
               'כל משחקי הכדורגל הפתוחים ב-Winner ובפולימרקט, עם יחסי Winner, הסתברויות ניצחון / תיקו / הפסד של '
               'המודל ותוחלת רווח. מתעדכן כל 3 שעות.'),
        sweep=('תופס פאניקה: קנייה במכירות פאניקה בפולימרקט',
               'בוט מסחר מדומה שמציב פקודות קנייה עמוקות לפני המשחק ותופס מכירות פאניקה בשווקי הכדורגל של '
               'פולימרקט: כל מילוי, מכירה ורווח.'),
        about=('איך מודלי ההימורים עובדים',
               'איך האתר עובד: יחסי Winner, מחירי פולימרקט, התאמת משחקים, חמשת המודלים ואיך הימורים מדומים '
               'נרשמים ומוכרעים.')),
}

HEBREW = re.compile('[֐-׿]')
OUTCOMES = {'Home Win': 0, 'Draw': 1, 'Away Win': 2}

TEXT = {
    'en': dict(dashboard='Dashboard', games='Games', about='About', sweep='Panic catcher', title='Paper trading on Polymarket',
               subtitle="Every 3 hours each model scores the open Winner games that are also on Polymarket. When a "
                        "model's probability beats the Polymarket price (taker fee included) by 3%, the outcome is "
                        "bought for 1 unit. Simulation only — no real money.",
               compare='Models compared', model='Model', bets='Bets', settled='Settled', won='Won', pnl='P&L',
               roi='ROI', clv='Avg CLV', open='Open', awaiting='Awaiting result', history='Settled bets',
               cum='Cumulative P&L (units)', min_ev='Min EV', all='All', kickoff='Kickoff', match='Match',
               pick='Pick', winner='Winner', poly='Polymarket', p='Model', ev='EV', bought='Bought', result='Result',
               running='Running', none_open='No open bets.', none_settled='No settled bets yet.',
               winner_picks='Picks at Winner odds', void='Void', draw='Draw', updated='Updated',
               clv_help="Closing-line value: the Polymarket price at kickoff vs what the bet cost. Above 0 over many "
                        "bets means the bets beat the market — it shows long before profit does.",
               win_rate='Win rate', avg_odds='Avg odds', table='Table'),
    'he': dict(dashboard='לוח בקרה', games='משחקים', about='אודות', sweep='תופס פאניקה', title='מסחר מדומה בפולימרקט',
               subtitle='כל 3 שעות כל מודל מעריך את משחקי Winner הפתוחים שקיימים גם בפולימרקט. כשההסתברות של המודל '
                        'עולה על מחיר פולימרקט (כולל עמלה) ב-3%, התוצאה "נקנית" ביחידה אחת. סימולציה בלבד — ללא כסף אמיתי.',
               compare='השוואת מודלים', model='מודל', bets='הימורים', settled='הוכרעו', won='זכו', pnl='רווח/הפסד',
               roi='תשואה', clv='CLV ממוצע', open='פתוחים', awaiting='ממתינים לתוצאה', history='הימורים שהוכרעו',
               cum='רווח/הפסד מצטבר (יחידות)', min_ev='EV מינימלי', all='הכל', kickoff='שעת פתיחה', match='משחק',
               pick='בחירה', winner='Winner', poly='פולימרקט', p='מודל', ev='EV', bought='נקנה', result='תוצאה',
               running='מצטבר', none_open='אין הימורים פתוחים.', none_settled='אין עדיין הימורים שהוכרעו.',
               winner_picks='בחירות ביחסי Winner', void='בוטל', draw='תיקו', updated='עודכן',
               clv_help='CLV: מחיר פולימרקט בשריקת הפתיחה לעומת מחיר הקנייה. מעל 0 לאורך הרבה הימורים = ניצחון על השוק.',
               win_rate='אחוז זכייה', avg_odds='יחס ממוצע', table='טבלה'),
}


def _lang():
    return 'he' if request.args.get('lang') in ('he', 'hebrew') else 'en'


def _name(hebrew, english, lang, names):
    """Display name: Hebrew as listed on Winner, or English (stored translation, else the name Polymarket uses)."""
    if lang == 'he':
        return hebrew or english or ''
    if english and not HEBREW.search(english):
        return english
    return names.get(hebrew) or english or hebrew or ''


def _float(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _row(p, lang, names):
    """One bet / pick in a common shape for the tables (paper bets, model picks and value-model picks)."""
    home = _name(p.get('home_team'), p.get('english_home_team'), lang, names)
    away = _name(p.get('away_team'), p.get('english_away_team'), lang, names)
    if lang == 'en' and ' vs. ' in (p.get('poly_title') or '') and HEBREW.search(home + away):
        home, away = p['poly_title'].split(' vs. ', 1)          # paper bets keep Polymarket's English title
    outcome = p.get('outcome', OUTCOMES.get(p.get('prediction')))
    flagged = p.get('flagged_at') or p.get('picked_at')
    if isinstance(flagged, str):
        try:
            flagged = datetime.fromisoformat(flagged[:16].replace('T', ' '))
        except ValueError:
            flagged = None
    pick = TEXT[lang]['draw'] if outcome == 1 else (home if outcome == 0 else away)
    if p.get('bet_no'):                                         # a bet against the outcome (longshot_v1)
        pick = f"{'לא' if lang == 'he' else 'Not'} {pick}"
    return dict(
        id=p.get('doc_id'), kickoff=p['match_time'], home=home, away=away,
        league=_name(p.get('league'), p.get('english_league'), lang, names), pick=pick,
        winner_odds=_float(p.get('winner_odds') or p.get('pick_odds')), poly_odds=_float(p.get('poly_odds')),
        poly_ask=_float(p.get('poly_ask')),
        poly_url=f"https://polymarket.com/event/{p['poly_slug']}" if p.get('poly_slug') else None,
        p_model=_float(p.get('p_model') or p.get('confidence')), ev=_float(p.get('ev', p.get('expected_value'))),
        bought=flagged, status=p.get('status') or 'pending', won=bool(p.get('prediction_result')),
        score=(f"{int(p['home_score'])}–{int(p['away_score'])}"
               if p.get('home_score') is not None and p.get('away_score') is not None else ''),
        profit=_float(p.get('profit')), clv=_float(p.get('clv')), odds=_float(p.get('pick_odds')))


def _summary(rows):
    """Split rows into open / awaiting / settled (newest first, with running P&L) and compute their stats."""
    now = datetime.now()
    settled = sorted((r for r in rows if r['status'] in ('completed', 'void')), key=lambda r: (r['kickoff'], r['id'] or ''))
    run = 0.0
    for r in settled:
        if r['profit'] is None:
            r['profit'] = 0.0 if r['status'] == 'void' else ((r['odds'] or 0) - 1 if r['won'] else -1.0)
        run += r['profit']
        r['running'] = run
    done = [r for r in settled if r['status'] == 'completed']
    n, clv = len(done), [r['clv'] for r in done if r['clv'] is not None]
    pending = [r for r in rows if r['status'] not in ('completed', 'void')]
    return dict(
        open=sorted((r for r in pending if r['kickoff'] > now), key=lambda r: r['kickoff']),
        awaiting=sorted((r for r in pending if r['kickoff'] <= now), key=lambda r: r['kickoff'], reverse=True),
        settled=settled[::-1],
        stats=dict(bets=len(rows), settled=n, won=sum(r['won'] for r in done), pnl=run,
                   roi=100 * run / n if n else None, clv=100 * sum(clv) / len(clv) if clv else None,
                   win_rate=100 * sum(r['won'] for r in done) / n if n else None,
                   avg_odds=sum(r['odds'] or 0 for r in done) / n if n else None,
                   open=sum(1 for r in pending if r['kickoff'] > now)))


def _min_ev():
    v = request.args.get('ev', request.args.get('ev_threshold'), type=float)
    return v if v in EV_OPTIONS else None


@app.route('/')
def dashboard():
    cache = get_cache()
    if not cache.ready:
        return _loading()
    lang, min_ev = _lang(), _min_ev()
    model = request.args.get('model') or request.args.get('model_type')
    model = model if model in MODELS else 'value_v1'
    names = cache.get('team_names', {})
    paper = [_row(p, lang, names) | dict(model=p.get('model_name')) for p in cache.get('paper_bets', [])]

    overview, chart = [], []
    for i, key in enumerate(MODELS):
        s = _summary([r for r in paper if r['model'] == key])
        overview.append(dict(key=key, label=MODELS[key]['label'], slot=i + 1, **s['stats']))
        points = [dict(x=r['kickoff'].strftime('%Y-%m-%dT%H:%M'), y=round(r['running'], 3)) for r in s['settled'][::-1]]
        chart.append(dict(key=key, label=MODELS[key]['label'], slot=i + 1, points=points))

    rows = [r for r in paper if r['model'] == model and (min_ev is None or (r['ev'] is not None and r['ev'] >= min_ev))]
    picks_key = MODELS[model]['picks']
    picks = _summary([_row(p, lang, names) for p in cache.get(picks_key, [])]) if picks_key else None
    last = max((p.get('prediction_timestamp') for p in cache.get('all_predictions', [])
                if isinstance(p.get('prediction_timestamp'), datetime)), default=None)
    return render_template('dashboard.html', t=TEXT[lang], lang=lang, models=MODELS, model=model, overview=overview,
                           chart=chart, sel=_summary(rows), picks=picks, min_ev=min_ev, ev_options=EV_OPTIONS,
                           updated=last, page='dashboard')


def _view(g, lang, names):
    return dict(id=g.get('id'), kickoff=g['match_time'], league_key=g.get('league'),
                league=_name(g.get('league'), g.get('english_league'), lang, names),
                home=_name(g.get('home_team'), g.get('english_home_team'), lang, names),
                away=_name(g.get('away_team'), g.get('english_away_team'), lang, names),
                odds=[(g.get('odds') or {}).get(k) for k in ('home', 'draw', 'away')],
                prob=[g.get(f'{k}_prob') for k in ('home_win', 'draw', 'away_win')],
                ev=[g.get(f'{k}_ev') for k in ('home_win', 'draw', 'away_win')],
                picked=[s in (g.get('flagged_sides') or []) for s in (0, 1, 2)],
                history=sorted(g.get('history') or [], key=lambda h: h.get('ts', ''), reverse=True))


def _loading():
    """Data not loaded yet (cold start): 503 + Retry-After so search engines come back instead of indexing it."""
    return render_template('loading.html'), 503, {'Retry-After': '30'}


@app.route('/game/<game_id>')
def game(game_id):
    cache = get_cache()
    if not cache.ready:
        return _loading()
    lang, names = _lang(), cache.get('team_names', {})
    from db import get_prediction
    items = cache.get('all_predictions', [])
    g = get_prediction(game_id) if any(p.get('id') == game_id for p in items) else None   # history: one read
    g = _view(g, lang, names) if g else None
    if g:
        date = g['kickoff'].strftime('%d %b %Y')
        if lang == 'he':
            seo = (f"{g['home']} נגד {g['away']}: תחזית ויחסים, {date}",
                   f"תחזית AI ל{g['home']} נגד {g['away']} ({g['league']}, {date}): יחסי Winner, הסתברויות ניצחון / "
                   f"תיקו / הפסד ותוחלת רווח, מתעדכן כל 3 שעות.")
        else:
            seo = (f"{g['home']} vs {g['away']} prediction and odds, {date}",
                   f"AI prediction for {g['home']} vs {g['away']} ({g['league']}, {date}): Winner odds, model win / "
                   f"draw / loss probabilities and expected value, updated every 3 hours.")
        start = g['kickoff'].replace(tzinfo=ISRAEL_TZ).isoformat()
        event = {'@context': 'https://schema.org', '@type': 'SportsEvent', 'name': f"{g['home']} vs {g['away']}",
                 'sport': 'Soccer', 'startDate': start, 'eventStatus': 'https://schema.org/EventScheduled',
                 'homeTeam': {'@type': 'SportsTeam', 'name': g['home']},
                 'awayTeam': {'@type': 'SportsTeam', 'name': g['away']},
                 'superEvent': {'@type': 'SportsEvent', 'name': g['league']} if g['league'] else None,
                 'url': SITE_URL + url_for('game', game_id=game_id, **({'lang': 'he'} if lang == 'he' else {}))}
        event = {k: v for k, v in event.items() if v is not None}
    else:
        seo, event = None, None
    html = render_template('game.html', t=TEXT[lang], lang=lang, g=g, game_id=game_id, page='games', seo=seo,
                           jsonld=event)
    return (html, 200) if g else (html, 404)     # a game leaves the list at kickoff: tell search engines it is gone


@app.route('/all')
def games():
    cache = get_cache()
    if not cache.ready:
        return _loading()
    lang, names = _lang(), cache.get('team_names', {})
    league, q, game_id = request.args.get('league', ''), request.args.get('q', '').strip(), request.args.get('game_id', '')
    items = cache.get('all_predictions', [])
    if game_id:                     # old links: the game page has its own URL now
        return redirect(url_for('game', game_id=game_id, **({'lang': 'he'} if lang == 'he' else {})), code=301)

    rows = sorted((_view(g, lang, names) for g in items), key=lambda r: r['kickoff'])
    leagues = sorted({(r['league_key'], r['league']) for r in rows if r['league_key']}, key=lambda x: x[1].lower())
    if league:
        rows = [r for r in rows if r['league_key'] == league]
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in (r['id'] or '').lower() or ql in r['home'].lower() or ql in r['away'].lower()]
    per_page, total = 50, len(rows)
    pages = max(1, -(-total // per_page))
    page_no = max(1, min(request.args.get('page', 1, type=int), pages))
    return render_template('games.html', t=TEXT[lang], lang=lang, rows=rows[(page_no - 1) * per_page: page_no * per_page],
                           total=total, picked=sum(any(r['picked']) for r in rows), leagues=leagues, league=league,
                           q=q, page_no=page_no, pages=pages, page='games')


ISRAEL = ZoneInfo('Asia/Jerusalem')


def _local(ts):
    """Unix seconds -> naive Israel time (the site shows Winner's local times)."""
    return datetime.fromtimestamp(ts, ISRAEL).replace(tzinfo=None) if ts else None


def _teams(title, home_he, away_he, lang):
    if lang == 'he' and home_he:
        return home_he, away_he
    if title and ' vs. ' in title:
        return tuple(title.split(' vs. ', 1))
    return home_he or '', away_he or ''


def _token_name(outcome, side, home, away, lang):
    name = (TEXT[lang]['draw'] if outcome == 1 else (home if outcome == 0 else away))
    return name if side > 0 else f"{'לא' if lang == 'he' else 'Not'} {name}"


@app.route('/sweep')
def sweep():
    """sweep_v1: deep resting buys that catch panic sales, paper-traded by the sweep_catch Lambda every 5 minutes."""
    cache = get_cache()
    if not cache.ready:
        return _loading()
    lang = _lang()
    data = cache.get('sweep', {}) or {}
    st = data.get('status') or {}
    rows = []
    for p in data.get('positions', []):
        home, away = _teams(p.get('poly_title'), p.get('home_team'), p.get('away_team'), lang)
        try:
            kick = datetime.strptime(p.get('match_time') or '', '%Y-%m-%d %H:%M')
        except ValueError:
            kick = _local(p.get('kick_ts'))
        stake, pnl = p.get('stake') or 0.0, p.get('pnl')
        rows.append(dict(id=p['doc_id'], kickoff=kick, home=home, away=away, status=p.get('status'), how=p.get('how'),
                         token=_token_name(p.get('outcome'), p.get('side', 1), home, away, lang), level=p.get('level'),
                         c=p.get('c'), L=p.get('L'), T=p.get('T'), q=p.get('q') or 0.0, q_exit=p.get('q_exit') or 0.0,
                         low=p.get('sweep_low'), filled=_local(p.get('fill_ts')), fill_ts=p.get('fill_ts') or 0,
                         mins=p.get('mins_to_kick'), stake=stake, pnl=pnl, liq=p.get('liq_price'),
                         ret=(100 * pnl / stake) if pnl is not None and stake else None,
                         q_sm=p.get('q_sm') or 0.0, stake_sm=p.get('stake_sm') or 0.0, pnl_sm=p.get('pnl_sm'),
                         url=f"https://polymarket.com/event/{p['poly_slug']}" if p.get('poly_slug') else None))
    closed = sorted((r for r in rows if r['status'] == 'closed'), key=lambda r: r['fill_ts'])
    run = run_sm = 0.0
    for r in closed:
        run += r['pnl'] or 0.0
        run_sm += r['pnl_sm'] or 0.0
        r['running'], r['running_sm'] = run, run_sm
    staked = sum(r['stake'] for r in closed)
    stats = dict(fills=len(rows), closed=len(closed), pnl=run, ret=100 * run / staked if staked else None,
                 staked=staked, target=sum(1 for r in closed if r['how'] == 'target'),
                 open=sum(1 for r in rows if r['status'] == 'open'), held=sum(1 for r in rows if r['status'] == 'held'))
    # the same trades at the size a $1,000 bankroll could rest (sweep.py: SM_BUDGET)
    small = [r for r in closed if r['q_sm'] > 0]
    staked_sm = sum(r['stake_sm'] for r in small)
    stats_sm = dict(pnl=run_sm, closed=len(small), staked=staked_sm, ret=100 * run_sm / staked_sm if staked_sm else None,
                    fills=sum(1 for r in rows if r['q_sm'] > 0), **(st.get('sm') or {}))
    live = []
    for o in st.get('live') or []:
        home, away = _teams(o.get('title'), o.get('home'), o.get('away'), lang)
        live.append(dict(kickoff=_local(o.get('kick_ts')), home=home, away=away, level=o.get('lvl'), c=o.get('c'),
                         L=o.get('L'), sm=o.get('sm') or 0.0, token=_token_name(o.get('o'), o.get('s', 1), home, away, lang)))
    last_run = st.get('last_run')
    last_run = _local(datetime.fromisoformat(last_run).timestamp()) if last_run else None
    points = [dict(x=r['filled'].strftime('%Y-%m-%dT%H:%M'), y=round(r['running'], 2)) for r in closed if r['filled']]
    points_sm = [dict(x=r['filled'].strftime('%Y-%m-%dT%H:%M'), y=round(r['running_sm'], 2)) for r in small if r['filled']]
    return render_template('sweep.html', t=TEXT[lang], lang=lang, stats=stats, small=stats_sm, points_sm=points_sm, status=st, last_run=last_run, live=live,
                           open=sorted((r for r in rows if r['status'] in ('open', 'held')), key=lambda r: -r['fill_ts']),
                           closed=closed[::-1], points=points, page='sweep')


@app.route('/about')
def about():
    lang = _lang()
    return render_template('about.html', t=TEXT[lang], lang=lang, models=MODELS, page='about')


@app.route('/health')
def health():
    """Also the keep-warm target (Cloud Scheduler, every 10 min): refreshes stale data so visitors don't wait."""
    get_cache()
    return {'status': 'ok'}


@app.route('/cache-status')
def cache_status():
    return get_cache().info()


@app.route('/sitemap.xml')
def sitemap():
    """Every public page in English and Hebrew (hreflang pairs), including a page per open game."""
    cache = get_cache()
    if not cache.ready:
        return Response('loading', 503, {'Retry-After': '30'})
    now = datetime.now(ISRAEL_TZ).strftime('%Y-%m-%d')
    pages = [(url_for(e), now, 'hourly' if e != 'about' else 'monthly') for e in ('dashboard', 'games', 'sweep', 'about')]
    for p in cache.get('all_predictions', []):
        if p.get('id'):
            ts = p.get('prediction_timestamp')
            pages.append((url_for('game', game_id=p['id']), ts.strftime('%Y-%m-%d') if ts else now, 'hourly'))
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">']
    for path, mod, freq in pages:
        en, he = SITE_URL + path, SITE_URL + path + '?lang=he'
        alts = (f'<xhtml:link rel="alternate" hreflang="en" href="{en}"/>'
                f'<xhtml:link rel="alternate" hreflang="he" href="{he}"/>'
                f'<xhtml:link rel="alternate" hreflang="x-default" href="{en}"/>')
        for loc in (en, he):
            out.append(f'<url><loc>{loc}</loc><lastmod>{mod}</lastmod><changefreq>{freq}</changefreq>{alts}</url>')
    out.append('</urlset>')
    return Response('\n'.join(out), mimetype='application/xml')


@app.template_filter('pct')
def pct(x, digits=1, signed=True):
    if x is None:
        return '–'
    return f"{x:+.{digits}f}%" if signed else f"{x:.{digits}f}%"


@app.template_filter('num')
def num(x, digits=2, signed=False):
    if x is None:
        return '–'
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


@app.template_filter('when')
def when(d, fmt='%d %b %H:%M'):
    return d.strftime(fmt) if d else '–'


@app.context_processor
def helpers():
    def url(**changes):
        """Current page URL with some query parameters changed (None removes one)."""
        args = request.args.to_dict()
        args.pop('model_type', None)
        args.pop('ev_threshold', None)
        args.update(changes)
        qs = urlencode({k: v for k, v in args.items() if v is not None and v != ''})
        return request.path + (f"?{qs}" if qs else '')
    def seo_links():
        """Canonical and language URLs of this page: only the path and the language count (filters, sorting and
        paging are views of the same page)."""
        en = SITE_URL + request.path
        he = en + '?lang=he'
        return dict(canonical=he if request.args.get('lang') in ('he', 'hebrew') else en, en=en, he=he)
    return dict(url=url, year=datetime.now().year, SEO=SEO, SITE_URL=SITE_URL, seo_links=seo_links)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8081)), debug=os.environ.get('FLASK_DEBUG') == '1')
