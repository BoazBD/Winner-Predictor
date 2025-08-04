from flask import Flask, render_template, send_from_directory, render_template_string, jsonify, request
from db import (
    # SQLite functions (will be conditionally used or replaced)
    get_profitable_games_from_db, 
    get_all_predictions_from_db, 
    get_unique_leagues_from_db,
    get_unique_models_from_db,
    get_paginated_predictions_from_db,
    # DynamoDB functions (will be conditionally used or replaced)
    get_profitable_games_from_dynamodb, 
    get_all_predictions_from_dynamodb,
    get_prediction_metadata_from_dynamodb,
    get_paginated_predictions_from_dynamodb,
    # Firestore functions (new)
    get_profitable_games_from_firestore,
    get_all_predictions_from_firestore,
    get_prediction_metadata_from_firestore,
    get_paginated_predictions_from_firestore,
    # Cache-aware functions (preferred)
    get_profitable_games_cached,
    get_all_predictions_cached,
    get_paginated_predictions_cached,
    get_prediction_metadata_cached,
    get_cache_status,
    # Cache functions that wait for initial data
    get_profitable_games_cached_with_wait,
    get_all_predictions_cached_with_wait,
    # Common or old
    DATA_SOURCE, # Import the DATA_SOURCE variable
    _firestore_doc_to_game_dict
)
import logging
from datetime import datetime, timedelta
import os
import time

# Translation dictionaries
TRANSLATIONS = {
    'english': {
        # Navigation
        'home': 'Home',
        'all_predictions': 'All Predictions',
        'about': 'About',
        'contact_us': 'Contact Us',
        'english': 'English',
        'hebrew': 'Hebrew',
        
        # Hero section
        'ai_powered_sports_predictions': 'AI-Powered Sports Predictions',
        'leverage_ml_algorithms': 'Leverage advanced machine learning algorithms to identify profitable betting opportunities.',
        'view_all_predictions': 'View All Predictions',
        'learn_more': 'Learn More',
        
        # Stats
        'win_rate': 'Win Rate',
        'average_roi': 'Average ROI',
        'successful_bets': 'Successful Bets',
        'total_predictions': 'Total Predictions',
        
        # Filters
        'model': 'Model',
        'all_models': 'All Models',
        'min_ev': 'Min EV',
        'league': 'League',
        'all_leagues': 'All Leagues',
        'status': 'Status',
        'all_statuses': 'All Statuses',
        'result': 'Result',
        'all_results': 'All Results',
        'prediction_type': 'Prediction Type',
        'all_predictions_types': 'All Prediction Types',
        'game_id': 'Game ID',
        
        # Game info
        'profitable_predictions': 'Profitable Predictions',
        'our_prediction': 'Our Prediction',
        'confidence': 'confidence',
        'details': 'Details',
        'outcome': 'Outcome',
        'probability': 'Probability',
        'odds': 'Odds',
        'expected_value': 'Expected Value',
        'profitable': 'Profitable',
        'home': 'Home',
        'draw': 'Draw',
        'away': 'Away',
        'final_score': 'Final Score',
        'result': 'Result',
        'result_id': 'Result ID',
        'predicted': 'Predicted',
        
        # Status badges
        'correct': 'Correct',
        'incorrect': 'Incorrect',
        'completed': 'Completed',
        'upcoming': 'Upcoming',
        'live': 'Live',
        
        # Pagination
        'previous': 'Previous',
        'next': 'Next',
        'page': 'Page',
        'of': 'of',
        'showing': 'Showing',
        'to': 'to',
        'entries': 'entries',
        
        # Errors and messages
        'no_profitable_predictions': 'No profitable predictions found for the selected model. Try another model or check back later.',
        'loading': 'Loading...',
        'search': 'Search',
        'clear_filters': 'Clear Filters',
        'apply_filters': 'Apply Filters',
        
        # Footer
        'all_rights_reserved': 'All rights reserved.',
        'version': 'Version',
        
        # About page
        'about_title': 'About AI Sport Better',
        'about_description': 'Advanced AI-powered sports betting predictions using machine learning.',
    },
    'hebrew': {
        # Navigation
        'home': 'בית',
        'all_predictions': 'כל התחזיות',
        'about': 'אודות',
        'contact_us': 'צור קשר',
        'english': 'אנגלית',
        'hebrew': 'עברית',
        
        # Hero section
        'ai_powered_sports_predictions': 'תחזיות ספורט מבוססות בינה מלאכותית',
        'leverage_ml_algorithms': 'נצל אלגוריתמי למידת מכונה מתקדמים כדי לזהות הזדמנויות הימורים רווחיות.',
        'view_all_predictions': 'צפה בכל התחזיות',
        'learn_more': 'למד עוד',
        
        # Stats
        'win_rate': 'אחוז זכייה',
        'average_roi': 'ROI ממוצע',
        'successful_bets': 'הימורים מוצלחים',
        'total_predictions': 'סך התחזיות',
        
        # Filters
        'model': 'מודל',
        'all_models': 'כל המודלים',
        'min_ev': 'EV מינימלי',
        'league': 'ליגה',
        'all_leagues': 'כל הליגות',
        'status': 'סטטוס',
        'all_statuses': 'כל הסטטוסים',
        'result': 'תוצאה',
        'all_results': 'כל התוצאות',
        'prediction_type': 'סוג תחזית',
        'all_predictions_types': 'כל סוגי התחזיות',
        'game_id': 'מזהה משחק',
        
        # Game info
        'profitable_predictions': 'תחזיות רווחיות',
        'our_prediction': 'התחזית שלנו',
        'confidence': 'ביטחון',
        'details': 'פרטים',
        'outcome': 'תוצאה',
        'probability': 'הסתברות',
        'odds': 'מקדמים',
        'expected_value': 'ערך צפוי',
        'profitable': 'רווחי',
        'home': 'בית',
        'draw': 'תיקו',
        'away': 'חוץ',
        'final_score': 'תוצאה סופית',
        'result': 'תוצאה',
        'result_id': 'מזהה תוצאה',
        'predicted': 'נחזה',
        
        # Status badges
        'correct': 'נכון',
        'incorrect': 'שגוי',
        'completed': 'הושלם',
        'upcoming': 'עתידי',
        'live': 'חי',
        
        # Pagination
        'previous': 'קודם',
        'next': 'הבא',
        'page': 'עמוד',
        'of': 'מתוך',
        'showing': 'מציג',
        'to': 'עד',
        'entries': 'רשומות',
        
        # Errors and messages
        'no_profitable_predictions': 'לא נמצאו תחזיות רווחיות עבור המודל הנבחר. נסה מודל אחר או חזור מאוחר יותר.',
        'loading': 'טוען...',
        'search': 'חיפוש',
        'clear_filters': 'נקה מסננים',
        'apply_filters': 'החל מסננים',
        
        # Footer
        'all_rights_reserved': 'כל הזכויות שמורות.',
        'version': 'גרסה',
        
        # About page
        'about_title': 'אודות AI Sport Better',
        'about_description': 'תחזיות הימורי ספורט מתקדמות מבוססות בינה מלאכותית באמצעות למידת מכונה.',
    }
}

def get_translation(key, language='english'):
    """Get translated text for a given key and language"""
    if language not in TRANSLATIONS:
        language = 'english'
    
    return TRANSLATIONS[language].get(key, TRANSLATIONS['english'].get(key, key))

def add_translations_to_context(language='english'):
    """Add all translations for a language to template context"""
    if language not in TRANSLATIONS:
        language = 'english'
    
    return TRANSLATIONS[language]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment from environment variables
FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
PORT = int(os.environ.get('PORT', 8081))
HOST = os.environ.get('HOST', '0.0.0.0')
DEBUG = os.environ.get('FLASK_DEBUG', '1') == '1'

# USE_DYNAMODB is now replaced by DATA_SOURCE from db.py
# USE_DYNAMODB = os.environ.get('USE_DYNAMODB', '1') == '1' 
logger.info(f"Application is using data source: {DATA_SOURCE}")

# DynamoDB tables - these might only be relevant if DATA_SOURCE can still be 'dynamodb' for some paths
ALL_PREDICTIONS_TABLE = "all-predicted-games" # Kept for reference if direct Dynamo calls were ever needed here
PROFITABLE_GAMES_TABLE = os.environ.get('PROFITABLE_GAMES_TABLE', 'profitable-games') # Kept for reference

app = Flask(__name__, static_folder='static')

# Initialize cache at startup
logger.info("Initializing cache system...")
try:
    from cache import initialize_cache
    cache = initialize_cache()
    logger.info("Cache system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize cache system: {e}")
    logger.warning("Falling back to direct database queries")

@app.route('/health')
def health_check():
    """Health check endpoint for Elastic Beanstalk"""
    # Optionally, add a check to see if db_firestore client is available if DATA_SOURCE is firestore
    db_status = "ok"
    if DATA_SOURCE == 'firestore':
        from db import db_firestore
        if db_firestore is None:
            db_status = "firestore_client_error"
            return jsonify({'status': 'unhealthy', 'reason': db_status, 'environment': FLASK_ENV}), 500

    return jsonify({'status': 'healthy', 'data_source': DATA_SOURCE, 'environment': FLASK_ENV}), 200

@app.route('/cache-status')
def cache_status():
    """Cache status endpoint for monitoring"""
    try:
        cache_info = get_cache_status()
        return jsonify({
            'status': 'ok',
            'cache_info': cache_info,
            'data_source': DATA_SOURCE
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'data_source': DATA_SOURCE
        }), 500

@app.route('/refresh-cache', methods=['POST'])
def refresh_cache():
    """Manual cache refresh endpoint for debugging"""
    try:
        from cache import get_cache
        cache = get_cache()
        cache.refresh_cache(force=True)
        return jsonify({
            'status': 'success',
            'message': 'Cache refreshed successfully',
            'cache_info': cache.get_cache_info()
        }), 200
    except Exception as e:
        logger.error(f"Error refreshing cache manually: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

# Sample games as fallback
SAMPLE_GAMES = [
    {
        'id': '1',
        'result_id': 'sample_result_1',
        'home_team': 'Team A',
        'away_team': 'Team B',
        'league': 'Premier League',
        'match_time': datetime.strptime('2024-04-05 20:00:00', '%Y-%m-%d %H:%M:%S'),
        'prediction': 'Home Win',
        'confidence': 0.85,
        'status': 'completed',
        'odds': {'home': 2.0, 'draw': 3.5, 'away': 4.0},
        'is_profitable': True,
        'expected_value': 0.35,
        'model_type': 'lstm_v2',
        'home_win_prob': 0.65,
        'draw_prob': 0.20,
        'away_win_prob': 0.15,
        'home_win_ev': 0.35,
        'draw_ev': -0.1,
        'away_win_ev': -0.2,
        'home_win_is_profitable': True,
        'draw_is_profitable': False,
        'away_win_is_profitable': False,
        'prediction_result': True,
        'actual_result': 'Home Win',
        'home_score': 2,
        'away_score': 0,
        'final_home_score': 2,
        'final_away_score': 0,
        'result_updated_at': datetime.now().isoformat()
    }
]

def ensure_game_has_required_fields(game):
    """Ensure a game object has all required fields for template rendering"""
    # Create a copy of the game to avoid modifying cached objects
    game = game.copy()
    
    # Required fields for basic rendering
    required_fields = {
        'id': 'unknown',
        'result_id': None,
        'home_team': 'Unknown Team',
        'away_team': 'Unknown Team',
        'league': 'Unknown League',
        'match_time': datetime.now(),
        'prediction': 'No prediction available',
        'confidence': None,
        'status': 'upcoming',
        'odds': {'home': None, 'draw': None, 'away': None},
        'expected_value': None,
        'is_profitable': None,
        'model_type': 'Not available',
        'home_win_prob': None,
        'draw_prob': None,
        'away_win_prob': None,
        'home_win_ev': None,
        'draw_ev': None,
        'away_win_ev': None,
        'home_win_is_profitable': None,
        'draw_is_profitable': None,
        'away_win_is_profitable': None,
        'prediction_result': None,
        'actual_result': None,
        'home_score': None,
        'away_score': None,
        'final_home_score': None,
        'final_away_score': None,
        'result_updated_at': None
    }
    
    # Add missing fields with default values (only for required structure)
    for field, default in required_fields.items():
        if field not in game:
            game[field] = default
    
    # Ensure match_time is a valid datetime object.
    # _firestore_doc_to_game_dict should have already parsed it from string or filtered the game.
    # SAMPLE_GAMES also provides match_time as datetime.
    if not isinstance(game['match_time'], datetime):
        logger.warning(f"Game ID {game.get('id', 'unknown')} 'match_time' is not a datetime object (type: {type(game['match_time'])}). Value: {game['match_time']}. Defaulting to current time.")
        game['match_time'] = datetime.now()
    
    # Ensure datetime is naive
    if game['match_time'].tzinfo is not None:
        game['match_time'] = game['match_time'].replace(tzinfo=None)

    # Add one minute to the match time to give users a buffer
    game['match_time'] = game['match_time'] + timedelta(minutes=1)
    
    # Make sure odds is a dictionary with the right keys, but don't add fictional odds
    if not isinstance(game['odds'], dict):
        game['odds'] = {'home': None, 'draw': None, 'away': None}
    else:
        for key in ['home', 'draw', 'away']:
            if key not in game['odds']:
                game['odds'][key] = None
    
    # Only normalize probabilities if all three probabilities are present and not None
    if game['home_win_prob'] is not None and game['draw_prob'] is not None and game['away_win_prob'] is not None:
        normalize_probabilities(game)
    
    return game

def normalize_probabilities(game):
    """Ensure that home_win_prob, draw_prob, and away_win_prob sum to exactly 1.0"""
    home_prob = game.get('home_win_prob', 0)
    draw_prob = game.get('draw_prob', 0)
    away_prob = game.get('away_win_prob', 0)
    
    # Calculate the sum of probabilities
    total_prob = home_prob + draw_prob + away_prob
    
    # Only normalize if the sum is greater than 0
    if total_prob > 0:
        game['home_win_prob'] = home_prob / total_prob
        game['draw_prob'] = draw_prob / total_prob
        game['away_win_prob'] = away_prob / total_prob

def calculate_prediction_stats(games):
    """Calculate prediction statistics from a list of games."""
    total_predictions = len(games)
    completed_games = [g for g in games if g.get('status') == 'completed']
    correct_predictions = [g for g in completed_games if g.get('prediction_result') == True]
    incorrect_predictions = [g for g in completed_games if g.get('prediction_result') == False]
    
    # Calculate win rate
    win_rate = len(correct_predictions) / len(completed_games) * 100 if completed_games else 0
    
    # Calculate average ROI
    total_return = 0
    total_investment = len(completed_games)  # Assuming 1 unit stake per game
    
    for game in completed_games:
        # Get the odds for the predicted outcome
        prediction = game.get('prediction', '')
        if prediction == 'Home Win':
            odds = game.get('odds', {}).get('home', 0)
        elif prediction == 'Draw':
            odds = game.get('odds', {}).get('draw', 0)
        elif prediction == 'Away Win':
            odds = game.get('odds', {}).get('away', 0)
        else:
            odds = 0
        
        if game.get('prediction_result') == True and odds > 0:
            # For winning bets, add the return (odds * stake)
            total_return += odds  # Assuming 1 unit stake
        # For losing bets, return is 0 (stake is lost)
    
    # ROI = (Total Return - Total Investment) / Total Investment * 100
    avg_roi = ((total_return - total_investment) / total_investment) * 100 if total_investment > 0 else 0
    
    return {
        'total_predictions': total_predictions,
        'win_rate': round(win_rate, 1),
        'successful_bets': len(correct_predictions),
        'unsuccessful_bets': len(incorrect_predictions),
        'avg_roi': round(avg_roi, 1)
    }

@app.route('/')
def index():
    try:
        # Get the selected model filter, default to 'lstm_100_12_v1'
        DEFAULT_MODEL = 'lstm_100_12_v1'
        selected_model = request.args.get('model_type', DEFAULT_MODEL) 
        
        # Get language preference, default to 'english'
        selected_language = request.args.get('lang', 'english')

        # Get EV threshold filter, default to 0
        ev_threshold_options = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05] # Representing 0%, 0.1%, ..., 1%, 2%, 3%, 4%, 5%
        selected_ev_threshold = request.args.get('ev_threshold', 0, type=float)
        if selected_ev_threshold not in ev_threshold_options: # Ensure valid threshold or default
            selected_ev_threshold = 0
        
        # Get profitable predictions from cache, waiting briefly for initial data
        games = get_profitable_games_cached()

        if not games:
            # If no games yet, render a loading page so the user doesn’t see an empty site
            logger.info("Cache not ready yet – showing loading page")
            return render_template('loading.html')
        
        # Ensure all games have required fields
        processed_games = [ensure_game_has_required_fields(game) for game in games]
        
        # Apply language preference if set to English
        if selected_language == 'english':
            for game in processed_games:
                if 'english_home_team' in game and game['english_home_team']:
                    game['display_home_team'] = game['english_home_team']
                else:
                    game['display_home_team'] = game['home_team']
                    
                if 'english_away_team' in game and game['english_away_team']:
                    game['display_away_team'] = game['english_away_team']
                else:
                    game['display_away_team'] = game['away_team']
                    
                if 'english_league' in game and game['english_league']:
                    game['display_league'] = game['english_league']
                else:
                    game['display_league'] = game['league']
        else:
            # Use original team and league names (Hebrew/original)
            for game in processed_games:
                game['display_home_team'] = game['home_team']
                game['display_away_team'] = game['away_team']
                game['display_league'] = game['league']
        
        # Get unique model names for the toggles
        model_types = sorted(list(set(game.get('model_name', 'Unknown Model') for game in processed_games)))
        
        # Apply model filter if selected
        if selected_model:
            processed_games = [game for game in processed_games if 
                              game.get('model_name') and selected_model.lower() in game.get('model_name', '').lower()]
            
            if len(processed_games) == 0:
                logger.warning(f"Filter '{selected_model}' resulted in zero games")
        
        # Apply EV threshold filter
        if selected_ev_threshold > 0:
            # Ensure 'expected_value' is present and is a float for comparison
            processed_games = [
                game for game in processed_games 
                if game.get('expected_value') is not None and float(game['expected_value']) >= selected_ev_threshold
            ]
        
        # Calculate statistics based on the displayed profitable games
        stats = calculate_prediction_stats(processed_games)
        
        # Get translations for the selected language
        translations = add_translations_to_context(selected_language)
        
        return render_template('index.html', 
                              games=processed_games, 
                              stats=stats,
                              model_types=model_types,
                              selected_model=selected_model,
                              selected_language=selected_language,
                              ev_threshold_options=ev_threshold_options,
                              selected_ev_threshold=selected_ev_threshold,
                              t=translations)
    
    except Exception as e:
        logger.error(f"Error fetching games for index: {str(e)}", exc_info=True)
        sample_games = [ensure_game_has_required_fields(game) for game in SAMPLE_GAMES]
        stats = calculate_prediction_stats(sample_games)
        # Also pass threshold options in case of error
        ev_threshold_options = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
        
        # Get translations for error case
        translations = add_translations_to_context('english')
        
        return render_template('index.html', 
                              games=sample_games, 
                              stats=stats,
                              model_types=['lstm_100_12_v1', 'lstm_100_12_v2'],
                              selected_model='',
                              selected_language='english',
                              ev_threshold_options=ev_threshold_options,
                              selected_ev_threshold=0,
                              t=translations)

@app.route('/all')
def all_predictions():
    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        league = request.args.get('league', '')
        model = request.args.get('model_type', '')
        game_id = request.args.get('game_id', '')
        
        # Get language preference
        language = request.args.get('lang', 'english')
        
        # Get predictions from Firestore
        try:
            games, total_count = get_paginated_predictions_cached(
                page=page,
                per_page=per_page,
                league=league,
                model=model,
                game_id=game_id
            )
        except Exception as e:
            if "requires a composite index" in str(e):
                # Get metadata for filters even in error case
                try:
                    leagues, models = get_prediction_metadata_cached()
                except:
                    leagues, models = [], []
                
                # Get translations for error case
                error_language = request.args.get('lang', 'english')
                translations = add_translations_to_context(error_language)
                
                return render_template(
                    'all_predictions.html',
                    games=[],
                    total_count=0,
                    page=page,
                    per_page=per_page,
                    total_pages=1,
                    leagues=leagues,
                    models=models,
                    model_types=models,  # Add this alias for template compatibility
                    selected_league=league,
                    selected_model=model,
                    selected_game_id=game_id,
                    game_id=game_id,
                    error_message=str(e),
                    language=language,
                    t=translations
                )
            else:
                # For other errors, re-raise
                raise e

        # Get unique leagues and models for filters
        leagues, models = get_prediction_metadata_cached()
        
        # Apply language preference to games (similar to index route)
        if language == 'english':
            for game in games:
                if 'english_home_team' in game and game['english_home_team']:
                    game['display_home_team'] = game['english_home_team']
                else:
                    game['display_home_team'] = game['home_team']
                    
                if 'english_away_team' in game and game['english_away_team']:
                    game['display_away_team'] = game['english_away_team']
                else:
                    game['display_away_team'] = game['away_team']
                    
                if 'english_league' in game and game['english_league']:
                    game['display_league'] = game['english_league']
                else:
                    game['display_league'] = game['league']
        else:
            # Use original team and league names
            for game in games:
                game['display_home_team'] = game['home_team']
                game['display_away_team'] = game['away_team']
                game['display_league'] = game['league']
        
        # Calculate pagination info
        total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
        
        # Get translations for the selected language
        translations = add_translations_to_context(language)
        
        return render_template(
            'all_predictions.html',
            games=games,
            total_count=total_count,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            leagues=leagues,
            models=models,
            model_types=models,  # Add this alias for template compatibility
            selected_league=league,
            selected_model=model,
            selected_game_id=game_id,
            game_id=game_id,
            selected_language=language,
            language=language,
            t=translations
        )
    except Exception as e:
        logger.error(f"Error in all_predictions route: {str(e)}", exc_info=True)
        
        # Get metadata for filters even in error case
        try:
            leagues, models = get_prediction_metadata_cached()
        except:
            leagues, models = [], []
        
        # Get translations for error case
        error_language = request.args.get('lang', 'english')
        translations = add_translations_to_context(error_language)
            
        return render_template(
            'all_predictions.html',
            games=[],
            total_count=0,
            page=1,
            per_page=50,
            total_pages=1,
            leagues=leagues,
            models=models,
            model_types=models,  # Add this alias for template compatibility
            selected_league='',
            selected_model='',
            selected_game_id='',
            game_id='',
            error_message=f"An error occurred: {str(e)}",
            language=error_language,
            t=translations
        )

@app.route('/about')
def about():
    # Get language preference
    selected_language = request.args.get('lang', 'english')
    translations = add_translations_to_context(selected_language)
    
    return render_template('about.html', 
                          selected_language=selected_language,
                          t=translations)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/robots.txt')
def robots_txt():
    """Serve robots.txt from the root path where crawlers expect it"""
    return send_from_directory(app.static_folder, 'robots.txt')

# API endpoints for the React frontend
@app.route('/api/profitable-predictions')
def api_profitable_predictions():
    try:
        # Get language preference, default to 'english'
        selected_language = request.args.get('lang', 'english')
        
        games = get_profitable_games_cached_with_wait(timeout_seconds=60)
        
        if not games:
            games = SAMPLE_GAMES.copy()
        else:
            # Sort by match time (newest first)
            games.sort(key=lambda x: x.get('match_time', datetime.min), reverse=True)
        
        # Ensure all games have required fields
        processed_games = [ensure_game_has_required_fields(game) for game in games]
        
        # Apply language preference if set to English
        if selected_language == 'english':
            for game in processed_games:
                if 'english_home_team' in game and game['english_home_team']:
                    game['display_home_team'] = game['english_home_team']
                else:
                    game['display_home_team'] = game['home_team']
                    
                if 'english_away_team' in game and game['english_away_team']:
                    game['display_away_team'] = game['english_away_team']
                else:
                    game['display_away_team'] = game['away_team']
                    
                if 'english_league' in game and game['english_league']:
                    game['display_league'] = game['english_league']
                else:
                    game['display_league'] = game['league']
        else:
            # Use original team and league names
            for game in processed_games:
                game['display_home_team'] = game['home_team']
                game['display_away_team'] = game['away_team']
                game['display_league'] = game['league']
        
        # Convert datetime objects to ISO format strings for JSON serialization
        for game in processed_games:
            if isinstance(game['match_time'], datetime):
                game['match_time'] = game['match_time'].isoformat()
        
        return jsonify({'predictions': processed_games, 'language': selected_language})
    
    except Exception as e:
        logger.error(f"Error fetching games for API: {str(e)}", exc_info=True)
        # Make sure sample games have all required fields
        sample_games = [ensure_game_has_required_fields(game) for game in SAMPLE_GAMES]
        
        # Convert datetime objects to ISO format strings
        for game in sample_games:
            if isinstance(game['match_time'], datetime):
                game['match_time'] = game['match_time'].isoformat()
        
        return jsonify({'predictions': sample_games, 'language': 'english'})

@app.route('/api/all-predictions')
def api_all_predictions():
    try:
        # Get language preference, default to 'english'
        selected_language = request.args.get('lang', 'english')
        
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        league = request.args.get('league', '')
        model = request.args.get('model', '')
        game_id = request.args.get('game_id', '')
        
        # Get predictions from cache
        games, total_count = get_paginated_predictions_cached(page, per_page, league, model, game_id)
        
        return jsonify({
            'games': games,
            'total': total_count,
            'page': page,
            'per_page': per_page
        })
    
    except Exception as e:
        logger.error(f"Error in API all-predictions: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'games': [],
            'total': 0,
            'page': page,
            'per_page': per_page
        }), 500

@app.route('/api/models')
def api_models():
    """API endpoint to list all available models"""
    try:
        games = get_all_predictions_cached_with_wait(timeout_seconds=60)

        # Get unique model names
        model_names = sorted(list(set(game.get('model_name', 'Unknown Model') for game in games)))
        
        # Count games by model
        model_counts = {}
        for model in model_names:
            model_counts[model] = len([game for game in games if game.get('model_name') == model])
        
        # Format response
        response = {
            'models': model_names,
            'model_counts': model_counts,
            'total_games': len(games)
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error fetching models for API: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def safely_parse_timestamp(timestamp_value):
    """Safely parse a timestamp value which might be a string or datetime object.
    Ensures the returned datetime is timezone-naive.
    """
    logger.info(f"Parsing timestamp: {timestamp_value} (type: {type(timestamp_value).__name__})")
    
    parsed_dt = None # Initialize variable to store the parsed datetime
    
    if isinstance(timestamp_value, datetime):
        logger.info(f"  -> Already a datetime object: {timestamp_value}")
        parsed_dt = timestamp_value
        
    elif isinstance(timestamp_value, str):
        try:
            # Handle ISO format timestamps like '2025-05-01T09:00:38.636543' or with timezone
            if 'T' in timestamp_value:
                # Replace Z with +00:00 for timezone handling if needed
                clean_timestamp = timestamp_value.replace('Z', '+00:00')
                parsed_dt = datetime.fromisoformat(clean_timestamp)
                logger.info(f"  -> Parsed ISO format: {parsed_dt}")
                
            # Handle simple date string formats
            elif ' ' in timestamp_value:  # Has date and time
                parsed_dt = datetime.strptime(timestamp_value, '%Y-%m-%d %H:%M:%S')
                logger.info(f"  -> Parsed datetime format: {parsed_dt}")
                
            else:  # Only date
                parsed_dt = datetime.strptime(timestamp_value, '%Y-%m-%d')
                logger.info(f"  -> Parsed date format: {parsed_dt}")
                
        except Exception as e:
            logger.warning(f"Error parsing timestamp '{timestamp_value}': {e}")
            parsed_dt = None # Ensure it's None on error
    
    # If parsing failed or input was not datetime/string, return fallback
    if parsed_dt is None:
        logger.warning(f"  -> Failed to parse timestamp or invalid type, using fallback: {datetime.min}")
        return datetime.min
        
    # --- Make the datetime timezone-naive --- 
    if parsed_dt.tzinfo is not None:
        logger.info(f"  -> Converting timezone-aware datetime {parsed_dt} to naive.")
        parsed_dt = parsed_dt.replace(tzinfo=None)
        logger.info(f"  -> Resulting naive datetime: {parsed_dt}")
    # --- End timezone conversion ---
    
    return parsed_dt

def get_sort_key_timestamp(game):
    """Extract the timestamp to use for sorting a game."""
    # --- Add logging inside this function --- 
    logger.debug(f"get_sort_key_timestamp called for game ID: {game.get('id')}, prediction_id: {game.get('prediction_id')}")
    val_to_parse = None
    source = "none"
    
    if 'prediction_timestamp' in game and game['prediction_timestamp']:
        val_to_parse = game['prediction_timestamp']
        source = "prediction_timestamp"
    elif 'match_time' in game and game['match_time']:
        val_to_parse = game['match_time']
        source = "match_time"
        
    if val_to_parse:
        logger.debug(f"  Attempting to parse '{source}': {val_to_parse} (type: {type(val_to_parse).__name__})")
        parsed_time = safely_parse_timestamp(val_to_parse) # Use the existing safe parser
        logger.debug(f"  Parsed result: {parsed_time} (type: {type(parsed_time).__name__})")
        return parsed_time
    else:
        logger.warning(f"  No suitable timestamp found (prediction_timestamp, match_time), returning datetime.min")
        return datetime.min
    # --- End added logging ---

def get_all_predictions_from_dynamodb():
    """Fetch all predictions from DynamoDB."""
    try:
        # Connect to DynamoDB
        dynamodb = boto3.resource('dynamodb', region_name="il-central-1")
        table = dynamodb.Table(ALL_PREDICTIONS_TABLE)
        
        logger.info(f"Fetching all predictions from DynamoDB table: {ALL_PREDICTIONS_TABLE}")
        
        # We'll use an index to fetch predictions efficiently
        # Using ModelTimeIndex for chronological order per model
        all_items = []
        
        # Configure retry logic to handle ProvisionedThroughputExceededException
        config = Config(
            retries = {
                'max_attempts': 10,
                'mode': 'adaptive'
            }
        )
        client = boto3.client('dynamodb', region_name="il-central-1", config=config)
        resource = boto3.resource('dynamodb', region_name="il-central-1", config=config)
        table = resource.Table(ALL_PREDICTIONS_TABLE)
        
        # First, get a list of unique model names by scanning the ModelTimeIndex
        # This is more efficient than scanning the entire table
        unique_models = set()
        try:
            # Use a smaller batch size to avoid throughput issues
            scan_params = {
                'ProjectionExpression': "model_name",
                'FilterExpression': boto3.dynamodb.conditions.Attr('model_name').exists(),
                'Limit': 25  # Use smaller batches
            }
            scan_response = table.scan(**scan_params)
            for item in scan_response.get('Items', []):
                if 'model_name' in item:
                    unique_models.add(item['model_name'])
            
            # Handle pagination if needed
            while 'LastEvaluatedKey' in scan_response:
                try:
                    # Add a small delay to avoid hitting throughput limits
                    time.sleep(0.5)
                    scan_params['ExclusiveStartKey'] = scan_response['LastEvaluatedKey']
                    scan_response = table.scan(**scan_params)
                    for item in scan_response.get('Items', []):
                        if 'model_name' in item:
                            unique_models.add(item['model_name'])
                except Exception as page_error:
                    logger.error(f"Error during pagination for model names: {str(page_error)}")
                    break  # Break the loop if we hit an error
        except Exception as e:
            logger.error(f"Error getting unique model names: {str(e)}")
            # Fall back to basic scan
            unique_models = []
        
        logger.info(f"Found {len(unique_models)} unique models in DynamoDB")
        
        # If we couldn't get unique models, fall back to a simple scan
        if not unique_models:
            logger.info("Falling back to simple scan for all items")
            try:
                # Use a smaller batch size and limit the number of items
                scan_params = {
                    'Limit': 50  # Get fewer items at a time
                }
                scan_response = table.scan(**scan_params)
                all_items = scan_response.get('Items', [])
                
                # Handle pagination if needed - limit to 5 pages to avoid throughput issues
                page_count = 1
                max_pages = 5
                while 'LastEvaluatedKey' in scan_response and page_count < max_pages:
                    try:
                        # Add a delay between scans
                        time.sleep(0.5)
                        scan_params['ExclusiveStartKey'] = scan_response['LastEvaluatedKey']
                        scan_response = table.scan(**scan_params)
                        all_items.extend(scan_response.get('Items', []))
                        page_count += 1
                    except Exception as page_error:
                        logger.error(f"Error during pagination for items: {str(page_error)}")
                        break
                
                logger.info(f"Fetched {len(all_items)} items in {page_count} pages")
            except Exception as scan_error:
                logger.error(f"Error during simple scan: {str(scan_error)}")
                all_items = []
        else:
            # Query each model separately using the GSI
            # Limit to first 3 models to avoid throughput issues
            for model_name in list(unique_models)[:3]:
                try:
                    # We're using the ModelTimeIndex GSI to get all predictions for a specific model
                    query_params = {
                        'IndexName': 'ModelTimeIndex',
                        'KeyConditionExpression': boto3.dynamodb.conditions.Key('model_name').eq(model_name),
                        'Limit': 50  # Limit items per query
                    }
                    query_response = table.query(**query_params)
                    
                    model_items = query_response.get('Items', [])
                    
                    # Handle pagination if needed - limit to 3 pages per model
                    page_count = 1
                    max_pages = 3
                    while 'LastEvaluatedKey' in query_response and page_count < max_pages:
                        try:
                            # Add a delay between queries
                            time.sleep(0.5)
                            query_params['ExclusiveStartKey'] = query_response['LastEvaluatedKey']
                            query_response = table.query(**query_params)
                            model_items.extend(query_response.get('Items', []))
                            page_count += 1
                        except Exception as page_error:
                            logger.error(f"Error during pagination for model {model_name}: {str(page_error)}")
                            break
                    
                    logger.info(f"Fetched {len(model_items)} items for model {model_name} in {page_count} pages")
                    all_items.extend(model_items)
                    
                except Exception as model_error:
                    logger.error(f"Error querying model {model_name}: {str(model_error)}")
                    # Continue with the next model
                    continue
        
        logger.info(f"Found {len(all_items)} total predictions in DynamoDB")
        
        # Process the items and format them for the template
        games = []
        for item in all_items:
            # Parse the prediction timestamp
            prediction_timestamp = item.get('timestamp', datetime.now().isoformat())
            if isinstance(prediction_timestamp, str):
                try:
                    prediction_timestamp = datetime.fromisoformat(prediction_timestamp)
                except:
                    prediction_timestamp = datetime.now()
                
            # Parse match time
            match_time_str = item.get('match_time_str')
            match_time = None
            if match_time_str:
                try:
                    match_time = datetime.strptime(match_time_str, '%Y-%m-%d %H:%M')
                except:
                    try:
                        match_time = datetime.fromisoformat(match_time_str)
                    except:
                        match_time = datetime.now()
            else:
                event_date = item.get('event_date')
                game_time = item.get('game_time')
                if event_date and game_time:
                    try:
                        match_time = datetime.strptime(f"{event_date} {game_time}", '%Y-%m-%d %H:%M')
                    except:
                        match_time = datetime.now()
                else:
                    match_time = datetime.now()
            
            # Determine the prediction based on EV or is_profitable flags
            home_win_ev = float(item.get('home_win_ev', 0))
            draw_ev = float(item.get('draw_ev', 0))
            away_win_ev = float(item.get('away_win_ev', 0))
            
            home_win_profitable = item.get('home_win_is_profitable', False)
            draw_profitable = item.get('draw_is_profitable', False)
            away_profitable = item.get('away_win_is_profitable', False)
            
            # Choose prediction based on expected value
            prediction = 'No prediction'
            highest_ev = max(home_win_ev, draw_ev, away_win_ev)
            if highest_ev == home_win_ev:
                prediction = 'Home Win'
            elif highest_ev == draw_ev:
                prediction = 'Draw'
            elif highest_ev == away_win_ev:
                prediction = 'Away Win'
            
            # Format the game dict
            game = {
                'id': item.get('game_id', item.get('id', 'unknown')),  # Use game_id or fall back to id
                'home_team': item.get('home_team', 'Unknown Team'),
                'away_team': item.get('away_team', 'Unknown Team'),
                'league': item.get('league', 'Unknown League'),
                # Add English versions of team names and league
                'english_home_team': item.get('english_home_team', ''),
                'english_away_team': item.get('english_away_team', ''),
                'english_league': item.get('english_league', ''),
                'match_time': match_time,
                'prediction': prediction,
                'confidence': max(float(item.get('home_win_prob', 0)), 
                                float(item.get('draw_prob', 0)), 
                                float(item.get('away_win_prob', 0))),
                'status': item.get('status', 'upcoming'),
                'odds': {
                    'home': float(item.get('home_odds', 0)),
                    'draw': float(item.get('draw_odds', 0)),
                    'away': float(item.get('away_odds', 0))
                },
                'is_profitable': item.get('is_profitable', 0) == 1,
                'expected_value': highest_ev,
                'model_name': item.get('model_name', 'Unknown Model'),
                'model_type': item.get('model_type', 'Unknown Type'),
                'home_win_prob': float(item.get('home_win_prob', 0)),
                'draw_prob': float(item.get('draw_prob', 0)),
                'away_win_prob': float(item.get('away_win_prob', 0)),
                'home_win_ev': home_win_ev,
                'draw_ev': draw_ev,
                'away_win_ev': away_win_ev,
                'home_win_is_profitable': home_win_profitable,
                'draw_is_profitable': draw_profitable,
                'away_win_is_profitable': away_profitable,
                'prediction_timestamp': prediction_timestamp,
                # Additional fields for completed games
                'prediction_result': item.get('prediction_result'),
                'actual_result': item.get('actual_result'),
                'home_score': item.get('home_score'),
                'away_score': item.get('away_score'),
                'final_home_score': item.get('final_home_score'),
                'final_away_score': item.get('final_away_score'),
                'result_updated_at': item.get('result_updated_at')
            }
            
            games.append(game)
        
        # Sort games by match time (upcoming first, then by confidence for same day)
        games.sort(key=lambda x: (x['match_time'], -x['confidence']))
        
        return games
    
    except Exception as e:
        logger.error(f"Error fetching all predictions from DynamoDB: {str(e)}", exc_info=True)
        return []

if __name__ == '__main__':
    app.run(debug=DEBUG, host=HOST, port=PORT) 