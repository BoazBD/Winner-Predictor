#!/usr/bin/env python3
import os
import json
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Import the Lambda function
import lambda_function

class TestLambdaFunction(unittest.TestCase):
    """Test cases for the Lambda function"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a simple test model
        self.create_test_model()
        
        # Create test data
        self.create_test_data()
        
        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'AWS_REGION': 'us-east-1',
            'ATHENA_DATABASE': 'test-db',
            'DYNAMODB_TABLE': 'test-table',
            'S3_BUCKET': 'test-bucket',
            'MODEL_TYPE': 'lstm',
            'EPOCHS': '100',
            'MAX_SEQ': '12',
            'THRESHOLD': '0.02'
        })
        self.env_patcher.start()
        
    def tearDown(self):
        """Clean up after tests"""
        self.env_patcher.stop()
        if os.path.exists('/tmp/lstm_100_12_v1.h5'):
            os.remove('/tmp/lstm_100_12_v1.h5')
    
    def create_test_model(self):
        """Create a simple test model"""
        # Define model architecture
        model = Sequential()
        model.add(LSTM(64, input_shape=(1, 9), return_sequences=False))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(3, activation='softmax')) # Output for home win, draw, away win probabilities
        
        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Save model to temp location
        model_path = '/tmp/lstm_100_12_v1.h5'
        model.save(model_path)
    
    def create_test_data(self):
        """Create test data for predictions"""
        # Create sample data with required columns
        self.test_data = pd.DataFrame({
            'game_id': ['game1', 'game2', 'game3'],
            'home_team': ['Team A', 'Team B', 'Team C'],
            'away_team': ['Team D', 'Team E', 'Team F'],
            'game_date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'home_win_odds': [2.0, 1.5, 3.0],
            'draw_odds': [3.5, 3.2, 3.8],
            'away_win_odds': [2.8, 5.0, 1.8],
            'home_team_rank': [5, 2, 10],
            'away_team_rank': [8, 15, 3],
            'home_goals_scored_avg': [1.5, 2.2, 0.8],
            'home_goals_conceded_avg': [0.8, 0.5, 1.2],
            'away_goals_scored_avg': [1.2, 0.7, 2.0],
            'away_goals_conceded_avg': [1.0, 1.8, 0.6]
        })
    
    @patch('lambda_function.s3')
    @patch('lambda_function.dynamodb')
    @patch('lambda_function.make_predictions')
    def test_lambda_handler(self, mock_make_predictions, mock_dynamodb, mock_s3):
        """Test the main Lambda handler function"""
        # Mock S3 download to use our local model
        mock_s3.download_file = MagicMock(return_value=None)
        
        # Mock DynamoDB table
        mock_table = MagicMock()
        mock_batch_writer = MagicMock()
        mock_table.batch_writer.return_value.__enter__.return_value = mock_batch_writer
        mock_dynamodb.Table.return_value = mock_table
        
        # Create some mock profitable predictions to test DynamoDB writes
        mock_profitable_games = [
            {
                'id': 'game1',
                'prediction': 'Home Win',
                'home_team': 'Team A',
                'away_team': 'Team D',
                'game_date': '2023-01-01',
                'odds': 2.0,
                'probability': 0.7,
                'expected_value': 0.4
            }
        ]
        
        # Mock the make_predictions function to return our profitable games
        mock_make_predictions.return_value = mock_profitable_games
        
        # Patch fetch_games_from_athena to return our test data
        with patch('lambda_function.fetch_games_from_athena', return_value=self.test_data):
            # Patch download_model_from_s3 to return our test model path
            with patch('lambda_function.download_model_from_s3', return_value='/tmp/lstm_100_12_v1.h5'):
                # Call the Lambda handler
                response = lambda_function.lambda_handler({}, {})
                
                # Assert the response
                self.assertEqual(response['statusCode'], 200)
                self.assertIn('Successfully processed', response['body'])
                
                # Verify DynamoDB was called
                mock_dynamodb.Table.assert_called_once()
                mock_table.batch_writer.assert_called_once()
                self.assertEqual(mock_batch_writer.put_item.call_count, 1)
                
        # Test case when no profitable predictions are found
        mock_make_predictions.return_value = []
        
        # Patch fetch_games_from_athena to return our test data
        with patch('lambda_function.fetch_games_from_athena', return_value=self.test_data):
            # Patch download_model_from_s3 to return our test model path
            with patch('lambda_function.download_model_from_s3', return_value='/tmp/lstm_100_12_v1.h5'):
                # Reset the mock objects
                mock_dynamodb.reset_mock()
                mock_table.reset_mock()
                
                # Call the Lambda handler
                response = lambda_function.lambda_handler({}, {})
                
                # Assert the response
                self.assertEqual(response['statusCode'], 200)
                self.assertIn('Successfully processed', response['body'])
                
                # Verify DynamoDB was NOT called when no profitable predictions
                mock_dynamodb.Table.assert_not_called()

def run_test():
    """Run the test suite"""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == '__main__':
    print("Running Lambda function tests...")
    run_test()
    print("Tests completed.") 