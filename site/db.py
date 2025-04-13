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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up AWS session
boto3.setup_default_session(region_name="il-central-1")
DATABASE = "winner-db"
ALL_PREDICTIONS_TABLE = os.environ.get('ALL_PREDICTIONS_TABLE', 'all-predicted-games')
PROFITABLE_GAMES_TABLE = os.environ.get('PROFITABLE_GAMES_TABLE', 'profitable-games')

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
    "ליגת האלופות האסיאתית",
    "ליגת האלופות האפריקאית",
    "גביע אנגלי",
    "גביע המדינה",
    "קונפרנס ליג",
    "מוקדמות מונדיאל, אסיה",
    "מוקדמות אליפות אירופה",
    "גרמנית ראשונה",
    "ליגת העל",
    "סופר קאפ",
    "ספרדית ראשונה",
    "ליגת האלופות",
    "הליגה האירופית",
    "גביע איטלקי",
    "ליגת האומות",
    "גביע המדינה Winner",
    "גביע הליגה האנגלי",
    "גביע אסיה",
]

# SQLite database path
DB_PATH = 'profitable_games.db'

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
    """Get profitable game predictions from DynamoDB"""
    try:
        # Create DynamoDB client
        dynamodb = boto3.resource('dynamodb', region_name="il-central-1")
        table = dynamodb.Table(PROFITABLE_GAMES_TABLE)
        
        logger.info(f"Fetching profitable games from DynamoDB table: {PROFITABLE_GAMES_TABLE}")
        
        # Since the table only contains profitable games, we don't need to filter
        response = table.scan()
        
        # Format the data for our application
        games = []
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
            
            # Create game object
            game = {
                'id': item.get('id', ''),
                'result_id': item.get('result_id', None),  # Include result_id field
                'home_team': item.get('home_team', ''),
                'away_team': item.get('away_team', ''),
                'league': item.get('league', ''),
                'match_time': match_time,
                'odds': odds,
                'status': item.get('status', 'pending'),
                'last_updated': item.get('timestamp', datetime.now().isoformat()),
                'model_type': f"{MODEL_TYPE}_v2",
                
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
                
                # For backward compatibility with templates
                'is_profitable': bool(item.get('home_win_is_profitable', False) or 
                                     item.get('draw_is_profitable', False) or 
                                     item.get('away_win_is_profitable', False)),
                
                # Find the highest EV and use it for sorting
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
                
                # Add result fields
                'prediction_result': item.get('prediction_result', None),
                'actual_result': item.get('actual_result', None),
                'home_score': item.get('home_score', None),
                'away_score': item.get('away_score', None),
                'final_home_score': item.get('final_home_score', None),
                'final_away_score': item.get('final_away_score', None),
                'result_updated_at': item.get('result_updated_at', None)
            }
            
            games.append(game)
        
        # Sort by expected value (highest first)
        games.sort(key=lambda x: x.get('expected_value', 0), reverse=True)
        
        logger.info(f"Retrieved {len(games)} profitable games from DynamoDB table {PROFITABLE_GAMES_TABLE}")
        return games
    except Exception as e:
        logger.error(f"Error retrieving games from DynamoDB: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_all_predictions_from_dynamodb():
    """Get all game predictions from DynamoDB"""
    try:
        # Create DynamoDB client
        dynamodb = boto3.resource('dynamodb', region_name="il-central-1")
        table = dynamodb.Table(ALL_PREDICTIONS_TABLE)
        
        logger.info(f"Fetching all predictions from DynamoDB table: {ALL_PREDICTIONS_TABLE}")
        
        # Scan the entire table
        response = table.scan()
        
        # Format the data for our application
        games = []
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
            
            # Create game object
            game = {
                'id': item.get('id', ''),
                'result_id': item.get('result_id', None),  # Include result_id field
                'home_team': item.get('home_team', ''),
                'away_team': item.get('away_team', ''),
                'league': item.get('league', ''),
                'match_time': match_time,
                'odds': odds,
                'status': item.get('status', 'pending'),
                'last_updated': item.get('timestamp', datetime.now().isoformat()),
                'model_type': f"{MODEL_TYPE}_{EPOCHS}_{MAX_SEQ}",
                
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
                
                # For backward compatibility with templates
                'is_profitable': bool(item.get('home_win_is_profitable', False) or 
                                     item.get('draw_is_profitable', False) or 
                                     item.get('away_win_is_profitable', False)),
                
                # Find the highest EV and use it for sorting
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
                
                # Add result fields
                'prediction_result': item.get('prediction_result', None),
                'actual_result': item.get('actual_result', None),
                'home_score': item.get('home_score', None),
                'away_score': item.get('away_score', None),
                'final_home_score': item.get('final_home_score', None),
                'final_away_score': item.get('final_away_score', None),
                'result_updated_at': item.get('result_updated_at', None)
            }
            
            games.append(game)
        
        # Sort by expected value (highest first)
        games.sort(key=lambda x: x.get('expected_value', 0), reverse=True)
        
        logger.info(f"Retrieved {len(games)} predictions from DynamoDB table {ALL_PREDICTIONS_TABLE}")
        return games
    except Exception as e:
        logger.error(f"Error retrieving predictions from DynamoDB: {e}")
        import traceback
        traceback.print_exc()
        return [] 
