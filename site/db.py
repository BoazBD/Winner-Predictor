import os
import pandas as pd
from datetime import datetime, timedelta, date
import boto3
import awswrangler as wr
import logging
import sqlite3
import json
import numpy as np
from decimal import Decimal
from boto3.dynamodb.conditions import Attr, Key
from botocore.config import Config
import time
from botocore.exceptions import ClientError
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Firestore Client Initialization
try:
    db_firestore = firestore.Client()
    logger.info("Firestore client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Firestore client: {e}")
    db_firestore = None

# Determine data source
DATA_SOURCE = os.environ.get('DATA_SOURCE', 'firestore').lower() # Default to firestore
logger.info(f"Data source configured to: {DATA_SOURCE}")

# Set up AWS session (kept for potential direct DynamoDB access or other AWS services)
boto3.setup_default_session(region_name="il-central-1")
DATABASE = "winner-db" # Athena database
ALL_PREDICTIONS_TABLE = os.environ.get('ALL_PREDICTIONS_TABLE', 'all-predicted-games') # DynamoDB table
PROFITABLE_GAMES_TABLE = os.environ.get('PROFITABLE_GAMES_TABLE', 'profitable-games') # DynamoDB table

# Firestore Collection Names
FIRESTORE_PREDICTIONS_COLLECTION = "predictions"
FIRESTORE_PROFITABLE_PREDICTIONS_COLLECTION = "profitable_predictions"

# Model configuration
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'lstm')
EPOCHS = os.environ.get('EPOCHS', '100')
MAX_SEQ = os.environ.get('MAX_SEQ', '12')

# List of big games leagues to filter by
BIG_GAMES = [
    "פרמייר ליג",
    "גביע ספרדי",
    "צרפתית ראשונה",
    "איטלקית ראשונה",
    "גביע אנגלי",
    "גביע המדינה",
    "קונפרנס ליג",
    "מוקדמות אליפות אירופה",
    "מוקדמות מונדיאל, אירופה"
    "גרמנית ראשונה",
    "ליגת העל",
    "סופר קאפ",
    "ספרדית ראשונה",
    "ליגת האלופות",
    "הליגה האירופית",
    "גביע איטלקי",
    "ליגת האומות",
    "גביע המדינה Winner",
    "ליגת Winner"
    "גביע הליגה האנגלי",
    "גביע אסיה",
    "גביע גרמני",
    "אליפות העולם לקבוצות",
    "גביע הטוטו",
]

# SQLite database path
DB_PATH = 'profitable_games.db'

def _firestore_doc_to_game_dict(doc_snapshot):
    """Converts a Firestore document snapshot to a game dictionary compatible with the app."""
    if not doc_snapshot.exists:
        return None
    game = doc_snapshot.to_dict()
    game['doc_id'] = doc_snapshot.id # Keep Firestore document ID (e.g., game_id_model_name)
    
    # Ensure the game's unique identifier is in game['id']
    # First try to get game_id from the document data
    if 'game_id' in game and game['game_id']:
        game['id'] = str(game['game_id'])
    elif 'prediction_source_id' in game and game['prediction_source_id']:
        game['id'] = str(game['prediction_source_id'])
    elif 'result_id' in game and game['result_id']:
        game['id'] = str(game['result_id'])
    else:
        # If no game_id in data, try to extract from doc_snapshot.id
        # Document ID format is typically game_id_model_name
        parts = doc_snapshot.id.split('_')
        if len(parts) > 1:
            game['id'] = str(parts[0])
        else:
            game['id'] = str(doc_snapshot.id)
            logger.warning(f"Could not determine specific game_id for doc {doc_snapshot.id}, using full doc_id as game['id']")

    # Convert Firestore Timestamps to datetime objects
    for key in ['prediction_timestamp', 'timestamp', 'result_updated_at']:
        if key in game and game[key] is not None:
            if isinstance(game[key], datetime): # Already datetime (e.g. from google.cloud.firestore.SERVER_TIMESTAMP)
                game[key] = game[key].replace(tzinfo=None) if game[key].tzinfo else game[key]
            else: # Assume it's a string or other type that needs parsing to datetime
                try:
                    # Firestore Timestamps are directly converted to datetime by to_dict()
                    # This path is more for data that might have been stored as strings initially by mistake
                    if isinstance(game[key], str):
                        game[key] = datetime.fromisoformat(game[key].replace('Z', '+00:00'))
                    # Ensure naive datetime
                    if hasattr(game[key], 'replace') and hasattr(game[key], 'tzinfo'):
                         game[key] = game[key].replace(tzinfo=None) if game[key].tzinfo else game[key]
                except (TypeError, ValueError) as e:
                    logger.error(f"Error converting timestamp field '{key}' for doc {doc_snapshot.id}: {e}. Value: {game[key]}")
                    game[key] = None # Or some default datetime
    
    # Convert match_time (stored as string YYYY-MM-DD HH:MM) to datetime object for app logic
    # The app.py ensure_game_has_required_fields expects datetime and adds timedelta.
    if 'match_time' in game and isinstance(game['match_time'], str):
        try:
            game['match_time'] = datetime.strptime(game['match_time'], '%Y-%m-%d %H:%M')
        except ValueError as e:
            logger.error(f"Error parsing match_time string for doc {doc_snapshot.id}: {e}. Value: {game['match_time']}. Game will be skipped.")
            return None # Skip game if match_time is unparsable
    elif 'match_time' not in game or game['match_time'] is None:
        logger.error(f"Missing or None match_time for doc {doc_snapshot.id}. Game will be skipped.")
        return None # Skip game if match_time is missing

    # Ensure numeric fields are floats (Firestore stores numbers as float64 or int64)
    numeric_fields = [
        'away_odds', 'away_win_ev', 'away_win_prob', 'draw_ev', 'draw_odds', 'draw_prob',
        'expected_value', 'home_odds', 'home_win_ev', 'home_win_prob', 'confidence',
        'home_score', 'away_score', 'final_home_score', 'final_away_score'
    ]
    for nf in numeric_fields:
        if nf in game and game[nf] is not None:
            try:
                game[nf] = float(game[nf])
            except (ValueError, TypeError):
                logger.warning(f"Could not convert numeric field {nf} to float for doc {doc_snapshot.id}. Value: {game[nf]}")
                game[nf] = 0.0 # Fallback

    # Ensure boolean fields are bool
    boolean_fields = [
        'away_win_is_profitable', 'draw_is_profitable', 'home_win_is_profitable', 'is_profitable', 'prediction_result'
    ]
    for bf in boolean_fields:
        if bf in game and game[bf] is not None:
            game[bf] = bool(game[bf])
    
    # Ensure `odds` is a dict with home, draw, away keys (similar to process_dynamodb_items)
    if 'odds' not in game or not isinstance(game['odds'], dict):
        game['odds'] = {'home': None, 'draw': None, 'away': None}
    else:
        for key in ['home', 'draw', 'away']:
            if key not in game['odds']:
                game['odds'][key] = None
            elif game['odds'][key] is not None:
                 game['odds'][key] = float(game['odds'][key])

    # Fallback for prediction and confidence if missing (similar to process_dynamodb_items)
    # This logic should ideally be handled when data is written to Firestore during migration/prediction saving.
    # However, as a safeguard for reading:
    if game.get('prediction') is None:
        hwev = game.get('home_win_ev', 0.0) or 0.0
        dev = game.get('draw_ev', 0.0) or 0.0
        awev = game.get('away_win_ev', 0.0) or 0.0
        if hwev >= dev and hwev >= awev:
            game['prediction'] = "Home Win"
        elif dev >= hwev and dev >= awev:
            game['prediction'] = "Draw"
        else:
            game['prediction'] = "Away Win"

    if game.get('confidence') is None:
        pred = game.get('prediction')
        if pred == "Home Win":
            game['confidence'] = float(game.get('home_win_prob', 0.0) or 0.0)
        elif pred == "Draw":
            game['confidence'] = float(game.get('draw_prob', 0.0) or 0.0)
        elif pred == "Away Win":
            game['confidence'] = float(game.get('away_win_prob', 0.0) or 0.0)
        else:
            game['confidence'] = 0.0
            
    if game.get('expected_value') is None: # Overall expected value
        game['expected_value'] = max(
            float(game.get('home_win_ev', 0.0) or 0.0),
            float(game.get('draw_ev', 0.0) or 0.0),
            float(game.get('away_win_ev', 0.0) or 0.0)
        )

    return game

def init_db():
    """Initialize the SQLite database for storing profitable games"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create table for profitable games if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS profitable_games (
            id TEXT PRIMARY KEY,
            home_team TEXT,
            away_team TEXT,
            league TEXT,
            match_time TEXT,
            prediction TEXT,
            confidence REAL,
            status TEXT,
            odds TEXT,
            expected_value REAL,
            is_profitable BOOLEAN,
            model_type TEXT,
            last_updated TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

def save_profitable_games(games):
    """Save all game predictions to the SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist with the new fields
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS profitable_games (
            id TEXT PRIMARY KEY,
            home_team TEXT,
            away_team TEXT,
            league TEXT,
            match_time TEXT,
            prediction TEXT,
            confidence REAL,
            status TEXT,
            odds TEXT,
            expected_value REAL,
            is_profitable BOOLEAN,
            model_type TEXT,
            last_updated TEXT
        )
        ''')
        
        for game in games:
            # Convert match_time to string for storage
            match_time_str = game['match_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(game['match_time'], datetime) else game['match_time']
            
            # Convert odds to JSON string
            odds_json = json.dumps(game['odds'])
            
            # Get is_profitable flag (default to False if not present)
            is_profitable = game.get('is_profitable', False)
            
            # Get model type if present
            model_type = game.get('model_type', 'unknown')
            
            # Insert or update the game
            cursor.execute('''
            INSERT OR REPLACE INTO profitable_games 
            (id, home_team, away_team, league, match_time, prediction, confidence, status, odds, expected_value, is_profitable, model_type, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                game['id'],
                game['home_team'],
                game['away_team'],
                game['league'],
                match_time_str,
                game['prediction'],
                game['confidence'],
                game['status'],
                odds_json,
                game.get('expected_value', 0.0),
                is_profitable,
                model_type,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(games)} game predictions to database")
    except Exception as e:
        logger.error(f"Error saving games to database: {e}")

def get_profitable_games_from_db():
    """Get only profitable games from the SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get only profitable games from the database
        cursor.execute('SELECT * FROM profitable_games WHERE is_profitable = 1')
        rows = cursor.fetchall()
        
        # Convert rows to game objects
        games = []
        for row in rows:
            # Parse match_time from string to datetime
            match_time = datetime.strptime(row[4], '%Y-%m-%d %H:%M:%S')
            
            # Parse odds from JSON string
            odds = json.loads(row[8])
            
            game = {
                'id': row[0],
                'home_team': row[1],
                'away_team': row[2],
                'league': row[3],
                'match_time': match_time,
                'prediction': row[5],
                'confidence': row[6],
                'status': row[7],
                'odds': odds,
                'expected_value': row[9],
                'is_profitable': bool(row[10]),
                'model_type': row[11] if len(row) > 11 else 'unknown',
                'last_updated': row[12] if len(row) > 12 else None
            }
            
            games.append(game)
        
        conn.close()
        logger.info(f"Retrieved {len(games)} profitable games from database")
        return games
    except Exception as e:
        logger.error(f"Error retrieving games from database: {e}")
        return []

def get_all_predictions_from_db():
    """Get all game predictions from the SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all games from the database
        cursor.execute('SELECT * FROM profitable_games')
        rows = cursor.fetchall()
        
        # Convert rows to game objects
        games = []
        for row in rows:
            # Parse match_time from string to datetime
            match_time = datetime.strptime(row[4], '%Y-%m-%d %H:%M:%S')
            
            # Parse odds from JSON string
            odds = json.loads(row[8])
            
            game = {
                'id': row[0],
                'home_team': row[1],
                'away_team': row[2],
                'league': row[3],
                'match_time': match_time,
                'prediction': row[5],
                'confidence': row[6],
                'status': row[7],
                'odds': odds,
                'expected_value': row[9],
                'is_profitable': bool(row[10]) if len(row) > 10 else False,
                'model_type': row[11] if len(row) > 11 else 'unknown',
                'last_updated': row[12] if len(row) > 12 else None
            }
            
            games.append(game)
        
        conn.close()
        logger.info(f"Retrieved {len(games)} game predictions from database")
        return games
    except Exception as e:
        logger.error(f"Error retrieving game predictions from database: {e}")
        return []

def get_latest_games_from_athena():
    """Fetch the latest games from AWS Athena"""
    logger.info("Fetching latest games from Athena")
    try:
        # Get the latest run_time in the dataset
        latest_run_time_query = "SELECT MAX(run_time) as latest_run_time FROM api_odds"
        latest_run_time_df = wr.athena.read_sql_query(latest_run_time_query, database=DATABASE)
        latest_run_time = latest_run_time_df['latest_run_time'].iloc[0]

        # Convert to datetime if it's a string
        if isinstance(latest_run_time, str):
            latest_run_time = pd.to_datetime(latest_run_time)

        logger.info(f"Latest run_time in Athena: {latest_run_time}")

        # Calculate date 7 days before the latest run_time
        seven_days_before = latest_run_time - timedelta(days=7)

        # Create a string of league names for the IN clause
        leagues_str = "', '".join(BIG_GAMES)

        # Query to get games from the last 7 days from the latest run_time
        query = f"""
        SELECT 
            *
        FROM 
            api_odds
        WHERE 
            date_parsed >= '{seven_days_before}'
            AND type = 'Soccer'
            AND league IN ('{leagues_str}')
        ORDER BY 
            run_time DESC
        """

        # Execute query and fetch results
        df = wr.athena.read_sql_query(query, database=DATABASE)

        # Convert run_time to datetime if it's not already
        df['run_time'] = pd.to_datetime(df['run_time'])

        # Get the latest run_time for each unique_id
        latest_times = df.groupby('unique_id')['run_time'].max()

        # Filter the dataframe to only include rows with the latest run_time for each unique_id
        latest_games = df[df.apply(lambda x: x['run_time'] == latest_times[x['unique_id']], axis=1)]

        # Process the data to match our application's format
        games = []
        for _, row in latest_games.iterrows():
            # Combine date and time for match_time and convert to datetime
            match_time_str = f"{row['event_date']} {row['time']}"
            match_time = datetime.strptime(match_time_str, '%Y-%m-%d %H:%M')

            # Create game object
            game = {
                'id': row['unique_id'],
                'home_team': row['option1'],
                'away_team': row['option3'],
                'league': row['league'],
                'match_time': match_time,
                'prediction': 'Pending',  # We don't have prediction data in the parquet
                'confidence': 0.5,  # Default confidence
                'status': 'pending',
                'odds': {
                    'home': float(row['ratio1']),
                    'draw': float(row['ratio2']),
                    'away': float(row['ratio3'])
                }
            }

            games.append(game)

        logger.info(f"Found {len(games)} games in Athena")
        return games

    except Exception as e:
        logger.error(f"Error fetching data from Athena: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_profitable_games_from_firestore():
    """Retrieves profitable game predictions from the Firestore profitable_predictions collection."""
    if not db_firestore:
        logger.error("Firestore client is not initialized in get_profitable_games_from_firestore. Cannot fetch profitable games.")
        return []
    logger.info(f"Firestore client appears to be initialized in get_profitable_games_from_firestore. Status: {db_firestore}")

    logger.info(f"Retrieving games from Firestore collection: {FIRESTORE_PROFITABLE_PREDICTIONS_COLLECTION}")
    profitable_games = []
    try:
        # Firestore stores datetime in UTC. Ensure `now` is also UTC for correct comparison if needed, 
        # but match_time is a string. For sorting, we convert match_time string to datetime.
        # The main sorting criteria relies on match_time and expected_value.
        
        # We want to order by match_time (desc for future, asc for past), and expected_value (desc)
        # Firestore querying capabilities for complex sorting on different fields with different directions can be tricky.
        # A common approach is to fetch and sort in Python if complex sorting isn't directly supported
        # or to add a composite field (e.g. future_flag + sortable_time).
        # For now, let's fetch all and sort in Python, similar to the DynamoDB version.
        # We can optimize with cursors/pagination later if performance is an issue.

        # Get all documents from the profitable_predictions collection
        # This could be a lot of data. For a production system, pagination is a must here.
        # The original DynamoDB function also fetched all and then sorted.
        docs_stream = db_firestore.collection(FIRESTORE_PROFITABLE_PREDICTIONS_COLLECTION).stream()
        
        raw_games = [_firestore_doc_to_game_dict(doc) for doc in docs_stream if doc.exists]
        
        # Filter out any None results from conversion (e.g., if a doc was malformed)
        raw_games = [g for g in raw_games if g is not None]

        logger.info(f"Found {len(raw_games)} raw games in Firestore profitable_predictions collection.")

        if not raw_games:
            return []

        # Get current time for separating past and future games (match_time is already datetime object here from _firestore_doc_to_game_dict)
        now = datetime.now()

        future_games = [game for game in raw_games if game.get('match_time') and game['match_time'] > now]
        past_games = [game for game in raw_games if game.get('match_time') and game['match_time'] <= now]

        logger.info(f"Found {len(future_games)} future profitable games from Firestore.")
        logger.info(f"Found {len(past_games)} past profitable games from Firestore.")

        # The original sorting was:
        # key=lambda x: (x['match_time'] > now, x['match_time'], -x['expected_value']), reverse=True
        # This means: 
        #   - Past games (False for `x['match_time'] > now`) come before Future games (True) because of reverse=True.
        #   - Within past games, later match_time comes first (due to reverse=True on x['match_time']).
        #   - Within future games, later match_time comes first.
        #   - Then by expected_value descending.
        # This is effectively sorting by match_time descending, and then by EV descending, but future games are not explicitly first.
        # The app.py index route shows upcoming games first.
        # Let's refine sort to ensure upcoming games are first, then sorted by newest, then by EV.

        # Correct sorting: Future games first, then by match_time descending, then by EV descending.
        # Past games after, then by match_time descending, then by EV descending.
        
        sorted_games = sorted(
            raw_games, 
            key=lambda x: (x['match_time'] <= now, # Puts future games (False) before past games (True)
                           -x['match_time'].timestamp() if x.get('match_time') else 0, # Sort by match_time descending (newest first)
                           -(x['expected_value'] or 0.0) # Sort by EV descending
                          )
        )       

        logger.info(f"Returning {len(sorted_games)} profitable games from Firestore, sorted.")
        return sorted_games

    except Exception as e:
        logger.error(f"Failed to retrieve profitable games from Firestore: {e}", exc_info=True)
        return []

def get_latest_games_from_parquet():
    """Fetch the latest games from the existing parquet file"""
    try:
        # Read the existing parquet file
        df = pd.read_parquet('latest_odds.parquet')
        
        # Convert run_time to datetime if it's not already
        df['run_time'] = pd.to_datetime(df['run_time'])
        
        # Get the latest run_time in the dataset
        latest_run_time = df['run_time'].max()
        logger.info(f"Latest run_time in parquet: {latest_run_time}")
        
        # Calculate date 7 days before the latest run_time
        seven_days_before = latest_run_time - timedelta(days=7)
        
        # Get the latest run_time for each unique_id
        latest_times = df.groupby('unique_id')['run_time'].max()
        
        # Filter the dataframe to only include rows with the latest run_time for each unique_id
        latest_games = df[df.apply(lambda x: x['run_time'] == latest_times[x['unique_id']], axis=1)]
        
        # Filter for games within 7 days of the latest run_time
        latest_games = latest_games[latest_games['run_time'] >= seven_days_before]
        
        # Filter for big games leagues
        latest_games = latest_games[latest_games['league'].isin(BIG_GAMES)]
        
        # Process the data to match our application's format
        games = []
        for _, row in latest_games.iterrows():
            # Combine date and time for match_time and convert to datetime
            match_time_str = f"{row['event_date']} {row['time']}"
            match_time = datetime.strptime(match_time_str, '%Y-%m-%d %H:%M')
            
            # Create game object
            game = {
                'id': row['unique_id'],
                'home_team': row['option1'],
                'away_team': row['option3'],
                'league': row['league'],
                'match_time': match_time,
                'prediction': 'Pending',  # We don't have prediction data in the parquet
                'confidence': 0.5,  # Default confidence
                'status': 'pending',
                'odds': {
                    'home': float(row['ratio1']),
                    'draw': float(row['ratio2']),
                    'away': float(row['ratio3'])
                }
            }
            
            games.append(game)
        
        logger.info(f"Found {len(games)} games in parquet file")
        return games
    
    except Exception as e:
        logger.error(f"Error reading parquet file: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_latest_games():
    """Get the latest games based on the environment"""
    # Initialize the database if it doesn't exist
    init_db()
    
    # Try to get games from the database first
    db_games = get_profitable_games_from_db()
    if db_games:
        logger.info("Using games from database")
        return db_games
    
    # If no games in database, fall back to the original methods
    logger.info("No games in database, falling back to original methods")
    if os.environ.get('FLASK_ENV') == 'production':
        return get_latest_games_from_athena()
    else:
        return get_latest_games_from_parquet()

def get_profitable_games_from_dynamodb():
    """
    Retrieves all profitable game predictions directly from the profitable-games DynamoDB table.
    
    Returns:
        list: A list of profitable game predictions, with past games grouped by game_id and model_name
              and future games kept individually (not grouped).
    """
    logger.info("Retrieving games directly from profitable-games table")
    
    # Configure retry logic to handle ProvisionedThroughputExceededException
    config = Config(
        retries = {
            'max_attempts': 10,
            'mode': 'adaptive'
        }
    )
    
    # Add region name explicitly to ensure connection works
    dynamodb = boto3.resource('dynamodb', region_name="il-central-1", config=config)
    table = dynamodb.Table(PROFITABLE_GAMES_TABLE)
    
    profitable_games = []
    
    try:
        # Scan the profitable-games table to get all games
        logger.info(f"Scanning {PROFITABLE_GAMES_TABLE} table")
        
        try:
            # Try to get the count of items, but this is optional
            count_response = table.scan(Select='COUNT', Limit=100)
            total_items = count_response.get('Count', 0)
            logger.info(f"Table {PROFITABLE_GAMES_TABLE} contains {total_items} total items (limited scan)")
        except Exception as e:
            logger.warning(f"Error counting items in DynamoDB table: {e}")
        
        # Scan the table for all items, with a smaller batch size
        scan_params = {
            'Limit': 50  # Limit items per scan
        }
        response = table.scan(**scan_params)
        items = response.get('Items', [])
        logger.info(f"Found {len(items)} games in profitable-games table")
        
        # Process items from the scan response
        profitable_games = process_dynamodb_items(items)
        
        # Handle pagination if needed, with limits
        page_count = 1
        max_pages = 5  # Limit the number of pages to avoid throughput issues
        while 'LastEvaluatedKey' in response and page_count < max_pages:
            try:
                # Add a small delay to avoid hitting throughput limits
                time.sleep(0.5)
                scan_params['ExclusiveStartKey'] = response['LastEvaluatedKey']
                response = table.scan(**scan_params)
                more_items = response.get('Items', [])
                logger.info(f"Found {len(more_items)} more games in profitable-games table (page {page_count+1})")
                profitable_games.extend(process_dynamodb_items(more_items))
                page_count += 1
            except Exception as page_error:
                logger.error(f"Error during pagination for profitable games: {str(page_error)}")
                break  # Stop pagination if we hit an error
                
    except Exception as e:
        logger.error(f"Failed to retrieve profitable games: {e}")
        return []
    
    # Get current time for separating past and future games
    now = datetime.now()
    
    # Separate future and past games
    future_games = [game for game in profitable_games if game['match_time'] > now]
    past_games = [game for game in profitable_games if game['match_time'] <= now]
    
    logger.info(f"Found {len(future_games)} future profitable games")
    logger.info(f"Found {len(past_games)} past profitable games")
    
    # Group past games by game_id and model_name
    # This ensures we don't show duplicate past games with the same model
    grouped_past_games = {}
    for game in past_games:
        key = (game['id'], game['model_name'])
        if key not in grouped_past_games or game['match_time'] > grouped_past_games[key]['match_time']:
            grouped_past_games[key] = game
    
    # Combine future games (not grouped) with past games (grouped)
    combined_games = future_games + list(grouped_past_games.values())
    
    # Sort by match time (future games first) and expected value (highest first)
    sorted_games = sorted(
        combined_games, 
        key=lambda x: (x['match_time'] > now, x['match_time'], -x['expected_value']),
        reverse=True  # Reverse the sort to put newest games first
    )
    
    logger.info(f"Returning {len(sorted_games)} profitable games in total")
    return sorted_games

def get_all_predictions_from_dynamodb():
    """Get all game predictions from DynamoDB"""
    try:
        # Create DynamoDB client
        dynamodb = boto3.resource('dynamodb', region_name="il-central-1")
        table = dynamodb.Table(ALL_PREDICTIONS_TABLE)
        
        logger.info(f"Fetching all predictions from DynamoDB table: {ALL_PREDICTIONS_TABLE}")
        
        # Scan the table to get all games
        response = table.scan()
        
        # Format the data for our application
        all_predictions = []
        for item in response.get('Items', []):
            # Parse match time 
            match_time = None
            
            # Try to get match_time from match_time_str first (newest format)
            if 'match_time_str' in item:
                try:
                    match_time = datetime.strptime(item['match_time_str'], '%Y-%m-%d %H:%M')
                except Exception as e:
                    logger.warning(f"Failed to parse match_time_str: {e}")
                    match_time = None
            
            # If match_time is still None, try to combine event_date and game_time
            if match_time is None and 'event_date' in item and 'game_time' in item:
                try:
                    match_time_str = f"{item['event_date']} {item['game_time']}"
                    match_time = datetime.strptime(match_time_str, '%Y-%m-%d %H:%M')
                except Exception as e:
                    logger.warning(f"Failed to parse event_date and game_time: {e}")
                    match_time = None
            
            # If match_time is still None, fall back to game_date (old format)
            if match_time is None and 'game_date' in item:
                try:
                    date_str = item['game_date']
                    # Check if game_date already has time component
                    if ' ' in date_str:
                        match_time = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                    else:
                        match_time = datetime.strptime(date_str, '%Y-%m-%d')
                except Exception as e:
                    logger.warning(f"Failed to parse game_date: {e}")
                    match_time = datetime.now()
            
            # Final fallback if match_time is still None
            if match_time is None:
                match_time = datetime.now()
                logger.warning("Using current time as fallback for match_time")
            
            # Format odds for display
            odds = {
                'home': float(item.get('home_odds', 0)),
                'draw': float(item.get('draw_odds', 0)),
                'away': float(item.get('away_odds', 0))
            }
            
            # Get the ID - could be either prediction_id + game_id (new schema) or just id (old schema)
            prediction_id = item.get('prediction_id', item.get('id', ''))
            game_id = item.get('game_id', item.get('id', ''))
            timestamp = item.get('timestamp', datetime.now().isoformat())
            
            # Get model information - make sure to include full model_name
            model_name = item.get('model_name', f"{MODEL_TYPE}_{EPOCHS}_{MAX_SEQ}_v1")
            
            # Process timestamp - make sure to set prediction_timestamp
            prediction_timestamp = item.get('prediction_timestamp', timestamp)
            
            # Create game object
            game = {
                'id': game_id,                   # Original game ID
                'prediction_id': prediction_id,  # Unique prediction ID
                'timestamp': timestamp,          # When the prediction was made
                'prediction_timestamp': prediction_timestamp,  # Consistent field for sorting
                'result_id': item.get('result_id', None),  # Include result_id field
                'home_team': item.get('home_team', ''),
                'away_team': item.get('away_team', ''),
                'league': item.get('league', ''),
                'match_time': match_time,
                'odds': odds,
                'status': item.get('status', 'pending'),
                'last_updated': item.get('timestamp', datetime.now().isoformat()),
                'model_name': model_name,  # Full model name
                'model_type': item.get('model_type', model_name.split('_')[0]),  # Model type (e.g., lstm) for compatibility
                
                # Model probabilities
                'home_win_prob': float(item.get('home_win_prob', 0.0)),
                'draw_prob': float(item.get('draw_prob', 0.0)),
                'away_win_prob': float(item.get('away_win_prob', 0.0)),
                
                # Expected values
                'home_win_ev': float(item.get('home_win_ev', 0.0)),
                'draw_ev': float(item.get('draw_ev', 0.0)),
                'away_win_ev': float(item.get('away_win_ev', 0.0)),
                
                # Profitable flags
                'home_win_is_profitable': bool(item.get('home_win_is_profitable', False)),
                'draw_is_profitable': bool(item.get('draw_is_profitable', False)),
                'away_win_is_profitable': bool(item.get('away_win_is_profitable', False)),
                
                # Overall profitable flag
                'is_profitable': bool(item.get('home_win_is_profitable', False) or 
                                     item.get('draw_is_profitable', False) or 
                                     item.get('away_win_is_profitable', False)),
                
                # Overall expected value (maximum of all outcomes) for sorting
                'expected_value': max(
                    float(item.get('home_win_ev', 0.0)),
                    float(item.get('draw_ev', 0.0)),
                    float(item.get('away_win_ev', 0.0))
                ),
                
                # For template compatibility - use highest EV outcome
                'prediction': (
                    "Home Win" if (float(item.get('home_win_ev', 0.0)) >= float(item.get('draw_ev', 0.0)) and 
                                float(item.get('home_win_ev', 0.0)) >= float(item.get('away_win_ev', 0.0))) else
                    "Draw" if (float(item.get('draw_ev', 0.0)) >= float(item.get('home_win_ev', 0.0)) and 
                            float(item.get('draw_ev', 0.0)) >= float(item.get('away_win_ev', 0.0))) else
                    "Away Win"
                ),
                'confidence': (
                    float(item.get('home_win_prob', 0.0)) if (float(item.get('home_win_ev', 0.0)) >= float(item.get('draw_ev', 0.0)) and 
                                                        float(item.get('home_win_ev', 0.0)) >= float(item.get('away_win_ev', 0.0))) else
                    float(item.get('draw_prob', 0.0)) if (float(item.get('draw_ev', 0.0)) >= float(item.get('home_win_ev', 0.0)) and 
                                                    float(item.get('draw_ev', 0.0)) >= float(item.get('away_win_ev', 0.0))) else
                    float(item.get('away_win_prob', 0.0))
                ),
                
                # Add result fields if available
                'prediction_result': item.get('prediction_result', None),
                'actual_result': item.get('actual_result', None),
                'home_score': item.get('home_score', None),
                'away_score': item.get('away_score', None),
                'final_home_score': item.get('final_home_score', None),
                'final_away_score': item.get('final_away_score', None),
                'result_updated_at': item.get('result_updated_at', None)
            }
            
            all_predictions.append(game)
        
        # Separate future and past games
        current_time = datetime.now()
        future_games = []
        past_games = []
        
        for game in all_predictions:
            if game.get('match_time', current_time) > current_time:
                future_games.append(game)
            else:
                past_games.append(game)
        
        # Group only the past games by game_id
        past_games_by_id = {}
        for game in past_games:
            game_id = game['id']
            timestamp = game.get('timestamp', '')
            
            if game_id not in past_games_by_id or timestamp > past_games_by_id[game_id].get('timestamp', ''):
                past_games_by_id[game_id] = game
            
        # Combine future games (ungrouped) and past games (grouped)
        model_predictions = future_games + list(past_games_by_id.values())
        
        # Sort by match time (newest first) then by expected value (highest first)
        if model_predictions:
            model_predictions.sort(key=lambda x: (
                # First sort by whether the game is in the future (upcoming games first)
                0 if x.get('match_time', current_time) > current_time else 1,
                # Then sort by match_time (newest first)
                -x.get('match_time', current_time).timestamp(),
                # Finally sort by expected value (highest first)
                -x.get('expected_value', 0)
            ))
        
        logger.info(f"Retrieved {len(model_predictions)} predictions for model {model_name} from DynamoDB ({len(future_games)} future, {len(past_games_by_id)} past)")
        return model_predictions
    except Exception as e:
        logger.error(f"Error retrieving game predictions from DynamoDB: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_predictions_by_model_from_dynamodb(model_name):
    """Get all game predictions for a specific model from DynamoDB"""
    try:
        # Create DynamoDB client
        dynamodb = boto3.resource('dynamodb', region_name="il-central-1")
        table = dynamodb.Table(ALL_PREDICTIONS_TABLE)
        
        logger.info(f"Fetching predictions for model {model_name} from DynamoDB table: {ALL_PREDICTIONS_TABLE}")
        
        # If model name contains underscores, like lstm_32_5, match the full name
        # If model name is a single term like 'lstm', match any model starting with that
        if '_' in model_name:
            # Exact match for full model name
            filter_expression = Attr('model_name').eq(model_name)
        else:
            # Prefix match for model type
            filter_expression = Attr('model_name').begins_with(model_name)
        
        # Scan the table with the filter
        response = table.scan(
            FilterExpression=filter_expression
        )
        
        # Format the data for our application
        all_predictions = []
        for item in response.get('Items', []):
            # Parse match time 
            match_time = None
            
            # Try to get match_time from match_time_str first (newest format)
            if 'match_time_str' in item:
                try:
                    match_time = datetime.strptime(item['match_time_str'], '%Y-%m-%d %H:%M')
                except Exception as e:
                    logger.warning(f"Failed to parse match_time_str: {e}")
                    match_time = None
            
            # If match_time is still None, try to combine event_date and game_time
            if match_time is None and 'event_date' in item and 'game_time' in item:
                try:
                    match_time_str = f"{item['event_date']} {item['game_time']}"
                    match_time = datetime.strptime(match_time_str, '%Y-%m-%d %H:%M')
                except Exception as e:
                    logger.warning(f"Failed to parse event_date and game_time: {e}")
                    match_time = None
            
            # If match_time is still None, fall back to game_date (old format)
            if match_time is None and 'game_date' in item:
                try:
                    date_str = item['game_date']
                    # Check if game_date already has time component
                    if ' ' in date_str:
                        match_time = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                    else:
                        match_time = datetime.strptime(date_str, '%Y-%m-%d')
                except Exception as e:
                    logger.warning(f"Failed to parse game_date: {e}")
                    match_time = datetime.now()
            
            # Final fallback if match_time is still None
            if match_time is None:
                match_time = datetime.now()
                logger.warning("Using current time as fallback for match_time")
            
            # Format odds for display
            odds = {
                'home': float(item.get('home_odds', 0)),
                'draw': float(item.get('draw_odds', 0)),
                'away': float(item.get('away_odds', 0))
            }
            
            # Get the ID - could be either prediction_id + game_id (new schema) or just id (old schema)
            prediction_id = item.get('prediction_id', item.get('id', ''))
            game_id = item.get('game_id', item.get('id', ''))
            timestamp = item.get('timestamp', datetime.now().isoformat())
            
            # Get model information - make sure to include full model_name
            item_model_name = item.get('model_name', f"{MODEL_TYPE}_{EPOCHS}_{MAX_SEQ}_v1")
            
            # Process timestamp - make sure to set prediction_timestamp
            prediction_timestamp = item.get('prediction_timestamp', timestamp)
            
            # Create game object
            game = {
                'id': game_id,                   # Original game ID
                'prediction_id': prediction_id,  # Unique prediction ID
                'timestamp': timestamp,          # When the prediction was made
                'prediction_timestamp': prediction_timestamp,  # Consistent field for sorting
                'result_id': item.get('result_id', None),  # Include result_id field
                'home_team': item.get('home_team', ''),
                'away_team': item.get('away_team', ''),
                'league': item.get('league', ''),
                'match_time': match_time,
                'odds': odds,
                'status': item.get('status', 'pending'),
                'last_updated': item.get('timestamp', datetime.now().isoformat()),
                'model_name': item_model_name,  # Full model name
                'model_type': item.get('model_type', item_model_name.split('_')[0]),  # Model type (e.g., lstm) for compatibility
                
                # Model probabilities
                'home_win_prob': float(item.get('home_win_prob', 0.0)),
                'draw_prob': float(item.get('draw_prob', 0.0)),
                'away_win_prob': float(item.get('away_win_prob', 0.0)),
                
                # Expected values
                'home_win_ev': float(item.get('home_win_ev', 0.0)),
                'draw_ev': float(item.get('draw_ev', 0.0)),
                'away_win_ev': float(item.get('away_win_ev', 0.0)),
                
                # Profitable flags
                'home_win_is_profitable': bool(item.get('home_win_is_profitable', False)),
                'draw_is_profitable': bool(item.get('draw_is_profitable', False)),
                'away_win_is_profitable': bool(item.get('away_win_is_profitable', False)),
                
                # Overall profitable flag
                'is_profitable': bool(item.get('home_win_is_profitable', False) or 
                                     item.get('draw_is_profitable', False) or 
                                     item.get('away_win_is_profitable', False)),
                
                # Overall expected value (maximum of all outcomes) for sorting
                'expected_value': max(
                    float(item.get('home_win_ev', 0.0)),
                    float(item.get('draw_ev', 0.0)),
                    float(item.get('away_win_ev', 0.0))
                ),
                
                # For template compatibility - use highest EV outcome
                'prediction': (
                    "Home Win" if (float(item.get('home_win_ev', 0.0)) >= float(item.get('draw_ev', 0.0)) and 
                                float(item.get('home_win_ev', 0.0)) >= float(item.get('away_win_ev', 0.0))) else
                    "Draw" if (float(item.get('draw_ev', 0.0)) >= float(item.get('home_win_ev', 0.0)) and 
                            float(item.get('draw_ev', 0.0)) >= float(item.get('away_win_ev', 0.0))) else
                    "Away Win"
                ),
                'confidence': (
                    float(item.get('home_win_prob', 0.0)) if (float(item.get('home_win_ev', 0.0)) >= float(item.get('draw_ev', 0.0)) and 
                                                        float(item.get('home_win_ev', 0.0)) >= float(item.get('away_win_ev', 0.0))) else
                    float(item.get('draw_prob', 0.0)) if (float(item.get('draw_ev', 0.0)) >= float(item.get('home_win_ev', 0.0)) and 
                                                    float(item.get('draw_ev', 0.0)) >= float(item.get('away_win_ev', 0.0))) else
                    float(item.get('away_win_prob', 0.0))
                ),
                
                # Add result fields if available
                'prediction_result': item.get('prediction_result', None),
                'actual_result': item.get('actual_result', None),
                'home_score': item.get('home_score', None),
                'away_score': item.get('away_score', None),
                'final_home_score': item.get('final_home_score', None),
                'final_away_score': item.get('final_away_score', None),
                'result_updated_at': item.get('result_updated_at', None)
            }
            
            all_predictions.append(game)
        
        # Separate future and past games
        current_time = datetime.now()
        future_games = []
        past_games = []
        
        for game in all_predictions:
            if game.get('match_time', current_time) > current_time:
                future_games.append(game)
            else:
                past_games.append(game)
        
        # Group only the past games by game_id
        past_games_by_id = {}
        for game in past_games:
            game_id = game['id']
            timestamp = game.get('timestamp', '')
            
            if game_id not in past_games_by_id or timestamp > past_games_by_id[game_id].get('timestamp', ''):
                past_games_by_id[game_id] = game
            
        # Combine future games (ungrouped) and past games (grouped)
        model_predictions = future_games + list(past_games_by_id.values())
        
        # Sort by match time (newest first) then by expected value (highest first)
        if model_predictions:
            model_predictions.sort(key=lambda x: (
                # First sort by whether the game is in the future (upcoming games first)
                0 if x.get('match_time', current_time) > current_time else 1,
                # Then sort by match_time (newest first)
                -x.get('match_time', current_time).timestamp(),
                # Finally sort by expected value (highest first)
                -x.get('expected_value', 0)
            ))
        
        logger.info(f"Retrieved {len(model_predictions)} predictions for model {model_name} from DynamoDB ({len(future_games)} future, {len(past_games_by_id)} past)")
        return model_predictions
    except Exception as e:
        logger.error(f"Error retrieving predictions for model {model_name} from DynamoDB: {e}")
        import traceback
        traceback.print_exc()
        return [] 

def process_dynamodb_items(items):
    """Process DynamoDB items into a format usable by the application"""
    processed_items = []
    
    for i, item in enumerate(items):        
        # Parse match time 
        match_time = None
        
        # Try to get match_time from match_time_str first (newest format)
        if 'match_time_str' in item:
            try:
                match_time = datetime.strptime(item['match_time_str'], '%Y-%m-%d %H:%M')
            except Exception as e:
                logger.error(f"Item {i+1}: Failed to parse match_time_str '{item.get('match_time_str')}': {e}")
                match_time = None
        
        # If match_time is still None, try to combine event_date and game_time
        if match_time is None and 'event_date' in item and 'game_time' in item:
            try:
                match_time_str = f"{item['event_date']} {item['game_time']}"
                match_time = datetime.strptime(match_time_str, '%Y-%m-%d %H:%M')
            except Exception as e:
                logger.error(f"Item {i+1}: Failed to parse event_date '{item.get('event_date')}' and game_time '{item.get('game_time')}': {e}")
                match_time = None
        
        # If match_time is still None, fall back to game_date (old format)
        if match_time is None and 'game_date' in item:
            try:
                date_str = item['game_date']
                # Check if game_date already has time component
                if ' ' in date_str:
                    match_time = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                else:
                    match_time = datetime.strptime(date_str, '%Y-%m-%d')
            except Exception as e:
                logger.error(f"Item {i+1}: Failed to parse game_date '{item.get('game_date')}': {e}")
                match_time = datetime.now()
        
        # Final fallback if match_time is still None
        if match_time is None:
            match_time = datetime.now()
            logger.warning(f"Item {i+1}: Using current time as fallback for match_time")
        
        # Format odds for display
        odds = {
            'home': float(item.get('home_odds', 0)),
            'draw': float(item.get('draw_odds', 0)),
            'away': float(item.get('away_odds', 0))
        }
        
        # Get the ID - could be either prediction_id + game_id (new schema) or just id (old schema)
        prediction_id = item.get('prediction_id', item.get('id', ''))
        game_id = item.get('game_id', item.get('id', ''))
        timestamp = item.get('timestamp', datetime.now().isoformat())
        
        # Get model information - make sure to include full model_name
        model_name = item.get('model_name', f"{MODEL_TYPE}_{EPOCHS}_{MAX_SEQ}_v1")
        
        # Process timestamp - make sure to set prediction_timestamp
        prediction_timestamp = item.get('prediction_timestamp', timestamp)
            
        # Create game object
        game = {
            'id': game_id,                   # Original game ID
            'prediction_id': prediction_id,  # Unique prediction ID
            'timestamp': timestamp,          # When the prediction was made
            'prediction_timestamp': prediction_timestamp,  # Consistent field for sorting
            'result_id': item.get('result_id', None),  # Include result_id field
            'home_team': item.get('home_team', ''),
            'away_team': item.get('away_team', ''),
            'league': item.get('league', ''),
            # Add the English version of team names and league
            'english_home_team': item.get('english_home_team', ''),
            'english_away_team': item.get('english_away_team', ''),
            'english_league': item.get('english_league', ''),
            'match_time': match_time,
            'odds': odds,
            'status': item.get('status', 'pending'),
            'last_updated': item.get('timestamp', datetime.now().isoformat()),
            'model_name': model_name,  # Full model name
            'model_type': item.get('model_type', model_name.split('_')[0]),  # Model type (e.g., lstm) for compatibility
            
            # Model probabilities
            'home_win_prob': float(item.get('home_win_prob', 0.0)),
            'draw_prob': float(item.get('draw_prob', 0.0)),
            'away_win_prob': float(item.get('away_win_prob', 0.0)),
            
            # Expected values
            'home_win_ev': float(item.get('home_win_ev', 0.0)),
            'draw_ev': float(item.get('draw_ev', 0.0)),
            'away_win_ev': float(item.get('away_win_ev', 0.0)),
            
            # Profitable flags
            'home_win_is_profitable': bool(item.get('home_win_is_profitable', False)),
            'draw_is_profitable': bool(item.get('draw_is_profitable', False)),
            'away_win_is_profitable': bool(item.get('away_win_is_profitable', False)),
            
            # Overall profitable flag
            'is_profitable': bool(item.get('home_win_is_profitable', False) or 
                                  item.get('draw_is_profitable', False) or 
                                  item.get('away_win_is_profitable', False)),
            
            # Overall expected value (maximum of all outcomes) for sorting
            'expected_value': max(
                float(item.get('home_win_ev', 0.0)),
                float(item.get('draw_ev', 0.0)),
                float(item.get('away_win_ev', 0.0))
            ),
            
            # For template compatibility - use highest EV outcome
            'prediction': (
                "Home Win" if (float(item.get('home_win_ev', 0.0)) >= float(item.get('draw_ev', 0.0)) and 
                            float(item.get('home_win_ev', 0.0)) >= float(item.get('away_win_ev', 0.0))) else
                "Draw" if (float(item.get('draw_ev', 0.0)) >= float(item.get('home_win_ev', 0.0)) and 
                        float(item.get('draw_ev', 0.0)) >= float(item.get('away_win_ev', 0.0))) else
                "Away Win"
            ),
            'confidence': (
                float(item.get('home_win_prob', 0.0)) if (float(item.get('home_win_ev', 0.0)) >= float(item.get('draw_ev', 0.0)) and 
                                                     float(item.get('home_win_ev', 0.0)) >= float(item.get('away_win_ev', 0.0))) else
                float(item.get('draw_prob', 0.0)) if (float(item.get('draw_ev', 0.0)) >= float(item.get('home_win_ev', 0.0)) and 
                                                 float(item.get('draw_ev', 0.0)) >= float(item.get('away_win_ev', 0.0))) else
                float(item.get('away_win_prob', 0.0))
            ),
            
            # Add result fields if available
            'prediction_result': item.get('prediction_result', None),
            'actual_result': item.get('actual_result', None),
            'home_score': item.get('home_score', None),
            'away_score': item.get('away_score', None),
            'final_home_score': item.get('final_home_score', None),
            'final_away_score': item.get('final_away_score', None),
            'result_updated_at': item.get('result_updated_at', None)
        }
        
        # --- Add detailed logging for processed game ---
        logger.info(f"Processed item {i+1}: Game Dict -> {game}")
        # --- End added logging ---
        
        processed_items.append(game)
    
    logger.info(f"--- Finished process_dynamodb_items ---") # Log end
    return processed_items 

# Add these new helper functions for paginated data retrieval

def get_unique_leagues_from_db():
    """Get a list of unique leagues from the database."""
    try:
        conn = sqlite3.connect('site/profitable_games.db')
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT league FROM predictions")
        leagues = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return sorted(leagues)
    except Exception as e:
        logging.error(f"Error getting unique leagues: {str(e)}")
        return []

def get_unique_models_from_db():
    """Get a list of unique model names from the database."""
    try:
        conn = sqlite3.connect('site/profitable_games.db')
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT model_name FROM predictions")
        models = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return sorted(models)
    except Exception as e:
        logging.error(f"Error getting unique models: {str(e)}")
        return []

def get_prediction_metadata_from_dynamodb():
    """Quickly retrieve just the metadata (leagues, models) from DynamoDB."""
    try:
        # Initialize empty lists
        leagues = []
        models = []
        
        # Connect to DynamoDB
        dynamodb = boto3.resource('dynamodb', region_name="il-central-1")
        table = dynamodb.Table('all-predicted-games')
        
        # Use a GSI to quickly get just the model names
        response = table.scan(
            ProjectionExpression="model_name, league",
            FilterExpression=Attr('model_name').exists() & Attr('league').exists(),
            Limit=100  # We just need a sample to get the metadata
        )
        
        # Extract unique models and leagues
        for item in response.get('Items', []):
            if 'model_name' in item and item['model_name'] not in models:
                models.append(item['model_name'])
            if 'league' in item and item['league'] not in leagues:
                leagues.append(item['league'])
        
        return sorted(leagues), sorted(models)
    except Exception as e:
        logging.error(f"Error fetching prediction metadata: {str(e)}")
        return [], []

def get_paginated_predictions_from_db(page, per_page, league='', model='', prediction='', status='', result='', ev='', game_id=''):
    """Get paginated predictions from SQLite with filters applied."""
    try:
        conn = sqlite3.connect('site/profitable_games.db')
        cursor = conn.cursor()
        
        # Start building the query
        base_query = "SELECT * FROM predictions WHERE 1=1"
        count_query = "SELECT COUNT(*) FROM predictions WHERE 1=1"
        params = []
        
        # Add filters
        if league:
            base_query += " AND league = ?"
            count_query += " AND league = ?"
            params.append(league)
            
        if game_id: # New filter for game_id
            base_query += " AND game_id = ?"
            count_query += " AND game_id = ?"
            params.append(game_id)
            
        if model:
            base_query += " AND model_name LIKE ?"
            count_query += " AND model_name LIKE ?"
            params.append(f"%{model}%")
            
        if prediction:
            base_query += " AND prediction = ?"
            count_query += " AND prediction = ?"
            params.append(prediction)
            
        if status:
            if status == 'upcoming':
                base_query += " AND (status = 'upcoming' OR status = 'pending')"
                count_query += " AND (status = 'upcoming' OR status = 'pending')"
            elif status == 'completed':
                base_query += " AND status = 'completed'"
                count_query += " AND status = 'completed'"
                
        if result:
            if result == 'correct':
                base_query += " AND prediction_result = 1"
                count_query += " AND prediction_result = 1"
            elif result == 'incorrect':
                base_query += " AND prediction_result = 0"
                count_query += " AND prediction_result = 0"
        
        if ev:
            if ev == 'high':
                base_query += " AND expected_value > 0.1"
                count_query += " AND expected_value > 0.1"
            elif ev == 'medium':
                base_query += " AND expected_value BETWEEN 0.05 AND 0.1"
                count_query += " AND expected_value BETWEEN 0.05 AND 0.1"
            elif ev == 'low':
                base_query += " AND expected_value BETWEEN 0 AND 0.05"
                count_query += " AND expected_value BETWEEN 0 AND 0.05"
                
        # Add ORDER BY and LIMIT for pagination
        offset = (page - 1) * per_page
        base_query += " ORDER BY prediction_timestamp DESC, match_time DESC LIMIT ? OFFSET ?"
        params.extend([per_page, offset])
        
        # Execute count query to get total
        cursor.execute(count_query, params[:-2] if params else [])
        total_count = cursor.fetchone()[0]
        
        # Execute main query
        cursor.execute(base_query, params)
        rows = cursor.fetchall()
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        
        # Convert to list of dictionaries
        games = []
        for row in rows:
            game = dict(zip(columns, row))
            
            # Parse JSON fields
            for field in ['odds', 'home_team_stats', 'away_team_stats']:
                if field in game and game[field]:
                    try:
                        game[field] = json.loads(game[field])
                    except:
                        game[field] = {}
            
            # Convert timestamp strings to datetime objects
            if 'match_time' in game and game['match_time']:
                try:
                    game['match_time'] = datetime.strptime(game['match_time'], '%Y-%m-%d %H:%M:%S')
                except:
                    pass
                    
            if 'prediction_timestamp' in game and game['prediction_timestamp']:
                try:
                    game['prediction_timestamp'] = datetime.strptime(game['prediction_timestamp'], '%Y-%m-%d %H:%M:%S')
                except:
                    pass
            
            games.append(game)
            
        conn.close()
        return games, total_count
    
    except Exception as e:
        logging.error(f"Error getting paginated predictions from DB: {str(e)}")
        return [], 0

def get_paginated_predictions_from_dynamodb(page, per_page, league='', model='', prediction='', status='', result='', ev='', game_id=''):
    """Get paginated predictions from DynamoDB. Uses Query on GSI if model is specified AND game_id is NOT. Query on table PK if game_id IS specified. Otherwise falls back to Scan."""
    
    dynamodb = boto3.resource('dynamodb', region_name="il-central-1")
    table = dynamodb.Table('all-predicted-games')

    filter_expressions = []
    expression_attr_values = {}
    expression_attr_names = {}

    # --- Helper to build filter expressions ---
    def add_filter(condition, attr_name_placeholder, attr_value_placeholder, value):
        filter_expressions.append(condition)
        expression_attr_values[attr_value_placeholder] = value
        # Ensure attr_name_placeholder starts with # before slicing
        if attr_name_placeholder.startswith('#'):
            expression_attr_names[attr_name_placeholder] = attr_name_placeholder[1:] # remove #
        else:
            # Handle cases where it might not start with #, though convention implies it should
            expression_attr_names[attr_name_placeholder] = attr_name_placeholder


    # --- Build FilterExpressions (common for Query and Scan) ---
    if league: add_filter('#league = :league', '#league', ':league', league)
    if prediction: add_filter('#prediction = :prediction', '#prediction', ':prediction', prediction)
    if status:
        if status == 'upcoming': add_filter('(#status = :upcoming OR #status = :pending)', '#status', ':upcoming', 'upcoming'); expression_attr_values[':pending'] = 'pending'
        elif status == 'completed': add_filter('#status = :completed', '#status', ':completed', 'completed')
    if result:
        prediction_result_val = True if result == 'correct' else False
        add_filter('#prediction_result = :result_val', '#prediction_result', ':result_val', prediction_result_val)
    if ev:
        try:
            # ev_val = Decimal(str(ev)) # Ensure EV is Decimal for comparison - not used directly in conditions below
            if ev == 'high': add_filter('#expected_value > :ev_val_high', '#expected_value', ':ev_val_high', Decimal('0.1'))
            elif ev == 'medium': add_filter('#expected_value BETWEEN :ev_min_medium AND :ev_max_medium', '#expected_value', ':ev_min_medium', Decimal('0.05')); expression_attr_values[':ev_max_medium'] = Decimal('0.1')
            elif ev == 'low': add_filter('#expected_value BETWEEN :ev_min_low AND :ev_max_low', '#expected_value', ':ev_min_low', Decimal('0.0')); expression_attr_values[':ev_max_low'] = Decimal('0.05')
        except Exception as e: logger.warning(f"Could not apply EV filter '{ev}': {e}")

    # --- Determine Query Strategy ---
    try:
        query_params = {} # Initialize query_params here
        if game_id: # Highest precedence: Query by game_id (PK)
            logger.info(f"Using DynamoDB Query on table PK for game_id: {game_id}")
            query_params = {
                'KeyConditionExpression': Key('game_id').eq(game_id),
                'ScanIndexForward': False,  # Sort by model_timestamp descending (if not filtered by specific model)
                'Limit': per_page
            }
            # If model is also specified, it becomes a filter expression for the PK query
            if model: add_filter('#model_name = :model', '#model_name', ':model', model)
        
        elif model: # Next: Query by model_name on GSI (ModelTimeIndex)
            logger.info(f"Using DynamoDB Query on GSI 'ModelTimeIndex' for model: {model}")
            query_params = {
                'IndexName': 'ModelTimeIndex',
                'KeyConditionExpression': Key('model_name').eq(model),
                'ScanIndexForward': False,  # Sort by timestamp (SK of GSI) descending
                'Limit': per_page
            }
            # game_id, if passed alongside model (but not as primary filter), would be a regular filter
            # This case is less likely now due to game_id taking precedence above.
        
        else: # Fallback to Scan (if neither game_id nor model is specified for Query)
            logger.info("Neither game_id nor model specified for Query, falling back to DynamoDB Scan.")
            # Scan implementation needs to be defined or use existing scan_fallback
            return get_paginated_predictions_from_dynamodb_scan_fallback(page, per_page, league, model, prediction, status, result, ev, game_id) # Pass game_id to scan too

        # Apply collected filters
        if filter_expressions:
            query_params['FilterExpression'] = ' AND '.join(filter_expressions)
            if expression_attr_values:
                for key, value in expression_attr_values.items(): # Ensure Decimals for numbers
                    if isinstance(value, float): expression_attr_values[key] = Decimal(str(value))
                query_params['ExpressionAttributeValues'] = expression_attr_values
            if expression_attr_names: query_params['ExpressionAttributeNames'] = expression_attr_names
            
        # --- Execute Query and Paginate --- 
        items = []
        last_evaluated_key = None
        pages_fetched = 0
        
        # Manual pagination loop to get to the desired page
        while pages_fetched < page:
            if last_evaluated_key: query_params['ExclusiveStartKey'] = last_evaluated_key
        
            response = table.query(**query_params)
            current_page_items = response.get('Items', [])
            
            if pages_fetched == page - 1: # This is the page we want
                items = current_page_items
            
            last_evaluated_key = response.get('LastEvaluatedKey')
            pages_fetched += 1
            if not last_evaluated_key: break # No more items to fetch

        # --- Get Total Count --- (Approximation for Query based on KeyCondition)
        # Create a new dict for count_query_params to avoid modifying query_params in place if it's used later
        count_query_params = {k: v for k, v in query_params.items() if k not in ['Limit', 'ExclusiveStartKey', 'ScanIndexForward']}
        count_query_params['Select'] = 'COUNT'
        total_count = 0
        try: 
            count_response = table.query(**count_query_params)
            total_count = count_response.get('Count', 0)
            # Handle potential pagination for count if FilterExpression is heavy
            temp_lek = count_response.get('LastEvaluatedKey')
            while temp_lek:
                count_query_params['ExclusiveStartKey'] = temp_lek
                count_response = table.query(**count_query_params)
                total_count += count_response.get('Count', 0)
                temp_lek = count_response.get('LastEvaluatedKey')
        except Exception as count_e:
            logger.error(f"Could not get exact count for DynamoDB query: {count_e}")

        logger.info(f"DynamoDB Query returned {len(items)} items for page {page}. Estimated total matching KeyCondition: {total_count}")
        games = process_dynamodb_items(items)
        return games, total_count 
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException' and 'IndexNotFound' in e.response['Error']['Message'] and model:
            logger.error(f"DynamoDB GSI 'ModelTimeIndex' not found for model query. Falling back to Scan.")
            return get_paginated_predictions_from_dynamodb_scan_fallback(page, per_page, league, model, prediction, status, result, ev, game_id)
        logger.error(f"Error during DynamoDB Query: {str(e)}", exc_info=True)
        return [], 0
    except Exception as e: # General exception catch
        logger.error(f"General error in get_paginated_predictions_from_dynamodb: {str(e)}", exc_info=True)
        return [], 0

# Fallback function using Scan (less reliable sorting for 'latest overall')
def get_paginated_predictions_from_dynamodb_scan_fallback(page, per_page, league='', model='', prediction='', status='', result='', ev='', game_id=''):
    logger.warning("Executing DynamoDB Scan fallback - results may not be perfectly sorted by latest overall timestamp until processed in app.")
    try:
        dynamodb = boto3.resource('dynamodb', region_name="il-central-1")
        table = dynamodb.Table('all-predicted-games')
        
        # Scan parameters
        scan_params = {}
        filter_expressions = []
        expression_attr_values = {}
        expression_attr_names = {}

        # Add filters (copied from Query version for consistency)
        if league: filter_expressions.append('#league = :league'); expression_attr_values[':league'] = league; expression_attr_names['#league'] = 'league'
        if game_id: filter_expressions.append('#game_id = :game_id'); expression_attr_values[':game_id'] = game_id; expression_attr_names['#game_id'] = 'game_id' # Add game_id filter
        if model: filter_expressions.append('contains(#model_name, :model)'); expression_attr_values[':model'] = model; expression_attr_names['#model_name'] = 'model_name'
        if prediction: filter_expressions.append('#prediction = :prediction'); expression_attr_values[':prediction'] = prediction; expression_attr_names['#prediction'] = 'prediction'
        if status:
            if status == 'upcoming': filter_expressions.append('(#status = :upcoming OR #status = :pending)'); expression_attr_values[':upcoming'] = 'upcoming'; expression_attr_values[':pending'] = 'pending'; expression_attr_names['#status'] = 'status'
            elif status == 'completed': filter_expressions.append('#status = :completed'); expression_attr_values[':completed'] = 'completed'; expression_attr_names['#status'] = 'status'
        if result:
            prediction_result_val = True if result == 'correct' else False
            filter_expressions.append('#prediction_result = :result_val'); expression_attr_values[':result_val'] = prediction_result_val; expression_attr_names['#prediction_result'] = 'prediction_result'
        if ev:
            try:
                if ev == 'high': filter_expressions.append('#expected_value > :high_ev'); expression_attr_values[':high_ev'] = Decimal('0.1')
                elif ev == 'medium': filter_expressions.append('#expected_value BETWEEN :medium_min AND :medium_max'); expression_attr_values[':medium_min'] = Decimal('0.05'); expression_attr_values[':medium_max'] = Decimal('0.1')
                elif ev == 'low': filter_expressions.append('#expected_value BETWEEN :low_min AND :low_max'); expression_attr_values[':low_min'] = Decimal('0.0'); expression_attr_values[':low_max'] = Decimal('0.05')
                if ev in ['high', 'medium', 'low']: expression_attr_names['#expected_value'] = 'expected_value'
            except Exception as e: logger.warning(f"Could not apply EV filter '{ev}': {e}")
        
        # Apply filters to scan parameters
        if filter_expressions:
            scan_params['FilterExpression'] = ' AND '.join(filter_expressions)
            if expression_attr_values:
                for key, value in expression_attr_values.items():
                    if isinstance(value, float): expression_attr_values[key] = Decimal(str(value))
                scan_params['ExpressionAttributeValues'] = expression_attr_values
            if expression_attr_names: scan_params['ExpressionAttributeNames'] = expression_attr_names

        # Manual pagination for Scan
        items = []
        last_evaluated_key = None
        items_needed = page * per_page 
        scan_limit = per_page # Adjust scan limit dynamically if needed, but start simple
        scan_params['Limit'] = scan_limit

        while len(items) < items_needed:
            if last_evaluated_key: scan_params['ExclusiveStartKey'] = last_evaluated_key
            
            try:
                response = table.scan(**scan_params)
            except ClientError as e:
                 logger.error(f"DynamoDB Scan Error: {e}")
                 return [], 0 # Return empty on scan error

            items.extend(response.get('Items', []))
            last_evaluated_key = response.get('LastEvaluatedKey')
            
            # Stop if no more items or if we fetched excessive amounts (safety break)
            if not last_evaluated_key or len(items) > items_needed + (per_page * 10): 
                break 
            # Simple throttle
            time.sleep(0.1) 
        
        # Get items for the specific page
        start_index = (page - 1) * per_page
        page_items = items[start_index : start_index + per_page] if start_index < len(items) else []
        games = process_dynamodb_items(page_items)
        
        # Get approximate total count matching filters (inefficient for large tables)
        count_params = {} 
        if scan_params.get('FilterExpression'): count_params['FilterExpression'] = scan_params['FilterExpression']
        if scan_params.get('ExpressionAttributeValues'): count_params['ExpressionAttributeValues'] = scan_params['ExpressionAttributeValues']
        if scan_params.get('ExpressionAttributeNames'): count_params['ExpressionAttributeNames'] = scan_params['ExpressionAttributeNames']
        count_params['Select'] = 'COUNT'
        
        total_count = 0
        try:
             count_response = table.scan(**count_params)
             total_count = count_response.get('Count', 0)
             # Handle count pagination if needed
             while 'LastEvaluatedKey' in count_response:
                 count_params['ExclusiveStartKey'] = count_response['LastEvaluatedKey']
                 count_response = table.scan(**count_params)
                 total_count += count_response.get('Count', 0)
        except Exception as count_e:
            logger.error(f"Could not get exact count from Scan: {count_e}")

        logger.info(f"DynamoDB Scan fallback returned {len(page_items)} items for page {page}. Total matching count: {total_count}")
        return games, total_count
        
    except Exception as e:
        logger.error(f"Error in DynamoDB Scan fallback: {str(e)}")
        import traceback; traceback.print_exc()
        return [], 0

def get_all_predictions_from_firestore():
    """Get all game predictions from the Firestore 'predictions' collection."""
    if not db_firestore:
        logger.error("Firestore client not initialized. Cannot fetch all predictions.")
        return []

    logger.info(f"Fetching all predictions from Firestore collection: {FIRESTORE_PREDICTIONS_COLLECTION}")
    all_predictions_processed = []
    try:
        # As with profitable games, fetching all documents can be resource-intensive.
        # Pagination and server-side filtering/sorting should be used in production for 'all predictions' view.
        # The original get_all_predictions_from_dynamodb also scanned all items then processed.
        docs_stream = db_firestore.collection(FIRESTORE_PREDICTIONS_COLLECTION).stream()
        
        raw_games = [_firestore_doc_to_game_dict(doc) for doc in docs_stream if doc.exists]
        all_predictions_processed = [g for g in raw_games if g is not None]

        if not all_predictions_processed:
            logger.info("No predictions found in Firestore.")
            return []

        # Replicate sorting logic from get_all_predictions_from_dynamodb
        # Separate future and past games
        current_time = datetime.now()
        future_games = []
        past_games = []
        
        for game in all_predictions_processed:
            game_match_time = game.get('match_time') # Already a datetime object
            if game_match_time and game_match_time > current_time:
                future_games.append(game)
            else:
                past_games.append(game)
        
        # Group only the past games by game_id (original DynamoDB version did this)
        # The doc ID in Firestore is game_id_model_name, so each (game_id, model_name) is unique.
        # If the intention of grouping was to show only the latest prediction for a game_id across different models,
        # then the Firestore query needs to be designed for that, or it happens in app.py.
        # The original get_all_predictions_from_dynamodb groups past_games by game_id, taking the one with the latest timestamp.
        # This implies multiple models could predict the same game_id, and we only want the latest past one.
        
        past_games_by_id = {}
        for game in past_games:
            game_id = game['id'] # This should be game_id
            # Use prediction_timestamp for sorting, fallback to datetime.min if not present
            game_prediction_ts = game.get('prediction_timestamp') or datetime.min
            
            current_game_in_dict = past_games_by_id.get(game_id)
            current_game_ts_in_dict = datetime.min
            if current_game_in_dict:
                # Use prediction_timestamp for existing game in dict, fallback to datetime.min
                current_game_ts_in_dict = current_game_in_dict.get('prediction_timestamp') or datetime.min

            if game_id not in past_games_by_id or game_prediction_ts > current_game_ts_in_dict:
                past_games_by_id[game_id] = game
            
        # Combine future games (ungrouped) and past games (grouped by game_id, latest prediction)
        model_predictions = future_games + list(past_games_by_id.values())
        
        # Sort by: 
        # 1. Future games first (is_future_flag == 0 for future, 1 for past)
        # 2. Then by match_time (descending - newest first)
        # 3. Finally by expected_value (descending - highest first)
        if model_predictions:
            for game in model_predictions: # Ensure sort keys are present and correct type
                 if not isinstance(game.get('match_time'), datetime):
                     game['match_time'] = current_time # fallback
                 game['expected_value'] = float(game.get('expected_value', 0.0) or 0.0)
            
            model_predictions.sort(key=lambda x: (
                x.get('match_time', current_time) <= current_time, # Future (False) before Past (True)
                -(x.get('match_time', current_time).timestamp()), # Newest match_time first
                -(x.get('expected_value', 0.0)) # Highest EV first
            ))
        
        logger.info(f"Retrieved {len(model_predictions)} predictions from Firestore ({len(future_games)} future, {len(past_games_by_id)} unique past grouped by game_id)")
        return model_predictions

    except Exception as e:
        logger.error(f"Error retrieving all game predictions from Firestore: {e}", exc_info=True)
        return []

def get_unique_models_from_db():
    """Get a list of unique model names from the database."""
    try:
        conn = sqlite3.connect('site/profitable_games.db')
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT model_name FROM predictions")
        models = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return sorted(models)
    except Exception as e:
        logging.error(f"Error getting unique models: {str(e)}")
        return []

def get_prediction_metadata_from_firestore():
    """Quickly retrieve unique leagues and model names from Firestore for filters."""
    if not db_firestore:
        logger.error("Firestore client not initialized. Cannot fetch metadata.")
        return [], []

    leagues = set()
    models = set()
    docs_processed_for_metadata = 0
    # Limit the number of documents scanned for metadata to avoid performance issues.
    # This provides a sample. For comprehensive metadata, a separate aggregation/metadata collection is better.
    METADATA_SCAN_LIMIT = 500 

    try:
        logger.info(f"Fetching metadata (leagues, models) from Firestore collection: {FIRESTORE_PREDICTIONS_COLLECTION} (limit {METADATA_SCAN_LIMIT})")
        docs_stream = db_firestore.collection(FIRESTORE_PREDICTIONS_COLLECTION).limit(METADATA_SCAN_LIMIT).stream()

        for doc in docs_stream:
            if doc.exists:
                data = doc.to_dict()
                if 'league' in data and data['league']:
                    leagues.add(data['league'])
                if 'model_name' in data and data['model_name']:
                    models.add(data['model_name'])
                docs_processed_for_metadata += 1
        
        logger.info(f"Processed {docs_processed_for_metadata} docs for metadata. Found {len(leagues)} unique leagues, {len(models)} unique models.")
        return sorted(list(leagues)), sorted(list(models))
    except Exception as e:
        logging.error(f"Error fetching prediction metadata from Firestore: {str(e)}", exc_info=True)
        return [], []

def get_paginated_predictions_from_firestore(page, per_page, league='', model='', game_id='', last_doc_snapshot=None):
    """Get paginated predictions from Firestore."""
    try:
        # Initialize Firestore client
        if not db_firestore:
            logger.error("Firestore client not initialized.")
            return [], 0
            
        predictions_ref = db_firestore.collection(FIRESTORE_PREDICTIONS_COLLECTION)
        
        logger.info(f"Querying Firestore for game_id: {game_id}, model: {model}, league: {league}")
        
        # Handle game_id filtering with document ID approach (most efficient)
        if game_id:
            logger.info(f"Filtering by game_id: {game_id}")
            # Get all documents and filter by document ID prefix
            all_docs = list(predictions_ref.stream())
            doc_id_prefix = f"{game_id}_"
            matching_docs = [doc for doc in all_docs if doc.id.startswith(doc_id_prefix)]
            logger.info(f"Found {len(matching_docs)} documents with ID prefix {doc_id_prefix}")
            
            # Convert documents to game dictionaries
            games = [_firestore_doc_to_game_dict(doc) for doc in matching_docs if doc.exists]
            games = [g for g in games if g is not None]  # Filter out None values
            
            # Apply additional filters if provided
            filtered_games = games
            
            # Apply league filter if provided
            if league:
                filtered_games = [g for g in filtered_games if g.get('league') == league]
                logger.info(f"Applied league filter '{league}' to game_id results, {len(filtered_games)} games remaining")
            
            # Apply model filter if provided
            if model:
                if '_' in model:
                    # Full model name match
                    filtered_games = [g for g in filtered_games if g.get('model_name') == model]
                else:
                    # Model type match
                    filtered_games = [g for g in filtered_games if g.get('model_type') == model]
                logger.info(f"Applied model filter '{model}' to game_id results, {len(filtered_games)} games remaining")
            
            # Sort by prediction_timestamp descending
            filtered_games.sort(key=lambda x: x.get('prediction_timestamp', datetime.min), reverse=True)
            
            return filtered_games, len(filtered_games)
        
        # For other filters, use a simpler approach to avoid composite index requirements
        # We'll fetch more documents and filter/sort client-side
        try:
            # Start with a simple query - no ordering to avoid composite index requirements
            query = predictions_ref
            
            # Apply only ONE field filter in Firestore to avoid composite index requirements
            # All other filters will be applied client-side
            firestore_filter_applied = None
            
            if league:
                query = query.where('league', '==', league)
                logger.info(f"Added Firestore league filter: {league}")
                firestore_filter_applied = 'league'
            elif model:
                # Only apply model filtering if no league filter (to avoid composite index)
                if '_' in model:
                    query = query.where('model_name', '==', model)
                else:
                    query = query.where('model_type', '==', model)
                logger.info(f"Added Firestore model filter: {model}")
                firestore_filter_applied = 'model'
            
            # Determine document limit based on filters
            if not firestore_filter_applied and not game_id:
                # No filters - get all documents
                docs = list(query.stream())
                logger.info(f"Retrieved all {len(docs)} documents from Firestore (no filters)")
            else:
                # Filters applied - use reasonable limit
                docs_limit = per_page * 50  # Increased limit for filtered queries to account for client-side filtering
                query = query.limit(docs_limit)
                docs = list(query.stream())
                logger.info(f"Found {len(docs)} documents from Firestore query with filters")
            
        except Exception as query_error:
            logger.warning(f"Firestore query failed, falling back to simple scan: {query_error}")
            # Fallback: Get all documents without any filters and filter client-side
            docs = list(predictions_ref.stream())
            logger.info(f"Fallback: Retrieved all {len(docs)} documents for client-side filtering")
        
        # Convert documents to game dictionaries
        games = []
        for doc in docs:
            if doc.exists:
                game_dict = _firestore_doc_to_game_dict(doc)
                if game_dict:
                    games.append(game_dict)
        
        logger.info(f"Converted {len(games)} documents to game dictionaries")
        
        # Apply client-side filtering for any filters that weren't applied in Firestore
        filtered_games = games
        
        # Apply league filter client-side if it wasn't applied in Firestore
        if league and firestore_filter_applied != 'league':
            filtered_games = [g for g in filtered_games if g.get('league') == league]
            logger.info(f"Client-side league filter applied, {len(filtered_games)} games remaining")
        
        # Apply model filter client-side if it wasn't applied in Firestore
        if model and firestore_filter_applied != 'model':
            if '_' in model:
                # Full model name match
                filtered_games = [g for g in filtered_games if g.get('model_name') == model]
            else:
                # Model type match
                filtered_games = [g for g in filtered_games if g.get('model_type') == model]
            logger.info(f"Client-side model filter applied, {len(filtered_games)} games remaining")
        
        # Sort by prediction_timestamp descending (client-side)
        filtered_games.sort(key=lambda x: x.get('prediction_timestamp', datetime.min), reverse=True)
        
        # Apply pagination (client-side)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_games = filtered_games[start_idx:end_idx]
        
        logger.info(f"Returning {len(paginated_games)} games for page {page} (out of {len(filtered_games)} total filtered games)")
        
        return paginated_games, len(filtered_games)
        
    except Exception as e:
        logger.error(f"Error querying Firestore: {str(e)}", exc_info=True)
        return [], 0

def get_prediction_metadata_from_dynamodb():
    """Quickly retrieve just the metadata (leagues, models) from DynamoDB."""
    try:
        # Initialize empty lists
        leagues = []
        models = []
        
        # Connect to DynamoDB
        dynamodb = boto3.resource('dynamodb', region_name="il-central-1")
        table = dynamodb.Table('all-predicted-games')
        
        # Use a GSI to quickly get just the model names
        response = table.scan(
            ProjectionExpression="model_name, league",
            FilterExpression=Attr('model_name').exists() & Attr('league').exists(),
            Limit=100  # We just need a sample to get the metadata
        )
        
        # Extract unique models and leagues
        for item in response.get('Items', []):
            if 'model_name' in item and item['model_name'] not in models:
                models.append(item['model_name'])
            if 'league' in item and item['league'] not in leagues:
                leagues.append(item['league'])
        
        return sorted(leagues), sorted(models)
    except Exception as e:
        logging.error(f"Error fetching prediction metadata: {str(e)}")
        return [], []

# Cache-aware functions that use local cache instead of direct database queries
def get_profitable_games_cached():
    """Get profitable games from local cache instead of database."""
    try:
        from cache import get_cache
        cache = get_cache()
        return cache.get_profitable_games()
    except Exception as e:
        logger.error(f"Error getting profitable games from cache, falling back to database: {e}")
        # Fallback to database if cache fails
        if DATA_SOURCE == 'firestore':
            return get_profitable_games_from_firestore()
        elif DATA_SOURCE == 'dynamodb':
            return get_profitable_games_from_dynamodb()
        elif DATA_SOURCE == 'sqlite':
            return get_profitable_games_from_db()
        else:
            return []

def get_all_predictions_cached():
    """Get all predictions from local cache instead of database."""
    try:
        from cache import get_cache
        cache = get_cache()
        return cache.get_all_predictions()
    except Exception as e:
        logger.error(f"Error getting all predictions from cache, falling back to database: {e}")
        # Fallback to database if cache fails
        if DATA_SOURCE == 'firestore':
            return get_all_predictions_from_firestore()
        elif DATA_SOURCE == 'dynamodb':
            return get_all_predictions_from_dynamodb()
        elif DATA_SOURCE == 'sqlite':
            return get_all_predictions_from_db()
        else:
            return []

def get_paginated_predictions_cached(page, per_page, league='', model='', game_id=''):
    """Get paginated predictions from local cache instead of database."""
    try:
        from cache import get_cache
        cache = get_cache()
        return cache.get_paginated_predictions(page, per_page, league, model, game_id)
    except Exception as e:
        logger.error(f"Error getting paginated predictions from cache, falling back to database: {e}")
        # Fallback to database if cache fails
        if DATA_SOURCE == 'firestore':
            return get_paginated_predictions_from_firestore(page, per_page, league, model, game_id)
        elif DATA_SOURCE == 'dynamodb':
            return get_paginated_predictions_from_dynamodb(page, per_page, league, model, '', '', '', '', game_id)
        elif DATA_SOURCE == 'sqlite':
            return get_paginated_predictions_from_db(page, per_page, league, model, '', '', '', '', game_id)
        else:
            return [], 0

def get_prediction_metadata_cached():
    """Get prediction metadata (leagues, models) from local cache instead of database."""
    try:
        from cache import get_cache
        cache = get_cache()
        metadata = cache.get_metadata()
        return metadata.get('leagues', []), metadata.get('models', [])
    except Exception as e:
        logger.error(f"Error getting metadata from cache, falling back to database: {e}")
        # Fallback to database if cache fails
        if DATA_SOURCE == 'firestore':
            return get_prediction_metadata_from_firestore()
        elif DATA_SOURCE == 'dynamodb':
            return get_prediction_metadata_from_dynamodb()
        else:
            return [], []

def get_cache_status():
    """Get cache status information for monitoring."""
    try:
        from cache import get_cache
        cache = get_cache()
        return cache.get_cache_info()
    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        return {
            'error': str(e),
            'cache_available': False
        }

# Cache functions that wait for initial data
def get_profitable_games_cached_with_wait(timeout_seconds: int = 60):
    """Get profitable games from local cache, waiting for initial data if needed."""
    try:
        from cache import get_cache
        cache = get_cache()
        return cache.get_profitable_games_with_wait(timeout_seconds)
    except Exception as e:
        logger.error(f"Error getting profitable games from cache with wait, falling back to database: {e}")
        # Fallback to database if cache fails
        if DATA_SOURCE == 'firestore':
            return get_profitable_games_from_firestore()
        elif DATA_SOURCE == 'dynamodb':
            return get_profitable_games_from_dynamodb()
        elif DATA_SOURCE == 'sqlite':
            return get_profitable_games_from_db()
        else:
            return []

def get_all_predictions_cached_with_wait(timeout_seconds: int = 60):
    """Get all predictions from local cache, waiting for initial data if needed."""
    try:
        from cache import get_cache
        cache = get_cache()
        return cache.get_all_predictions_with_wait(timeout_seconds)
    except Exception as e:
        logger.error(f"Error getting all predictions from cache with wait, falling back to database: {e}")
        # Fallback to database if cache fails
        if DATA_SOURCE == 'firestore':
            return get_all_predictions_from_firestore()
        elif DATA_SOURCE == 'dynamodb':
            return get_all_predictions_from_dynamodb()
        elif DATA_SOURCE == 'sqlite':
            return get_all_predictions_from_db()
        else:
            return []
