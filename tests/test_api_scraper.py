import json
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import requests

from api_request.api_scraper import (API_URL, HASH_CHECKSUM_URL, create_bet,
                                     fetch_data, headers, process_data,
                                     save_to_s3)


class TestApiScraper(unittest.TestCase):
    @patch("requests.post")
    def test_fetch_data_success(self, mock_get):
        mock_response = MagicMock()

        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_get.return_value = mock_response

        result = fetch_data(API_URL, lineChecksum="test")
        self.assertEqual(result, {"data": "test"})

    def test_process_data(self):
        with open("tests/test_response.json") as file:
            data = json.load(file)

        result = process_data(data)
        self.assertIsInstance(result, pd.DataFrame)

        self.assertEqual(result.shape[0], 497)
        self.assertEqual(result.shape[1], 11)

        not_nullable_columns = result.columns.difference(["option3", "ratio3"])
        self.assertTrue(result[not_nullable_columns].notnull().all().all())

        result["ratio1"] = result["ratio1"].astype(float)
        result["ratio2"] = result["ratio2"].astype(float)
        result["ratio3"] = result["ratio3"].astype(float)

        self.assertTrue((result["ratio1"] >= 1).all())
        self.assertTrue((result["ratio1"] <= 200).all())

        self.assertTrue((result["ratio2"] >= 1).all())
        self.assertTrue((result["ratio2"] <= 200).all())

        # Check that option3 and ratio3 are not null when the type is Soccer
        soccer_rows = result[result["type"] == "Soccer"]
        self.assertTrue(soccer_rows[["option3", "ratio3"]].notnull().all().all())

        self.assertTrue((soccer_rows["ratio3"] >= 1).all())
        self.assertTrue((soccer_rows["ratio3"] <= 200).all())

    def test_create_bet(self):
        market = {
            "sId": 240,
            "e_date": "240101",
            "m_hour": 1230,
            "mp": "Market",
            "league": "League",
            "outcomes": [
                {"desc": "Option 1", "price": 1.5},
                {"desc": "Option 2", "price": 2.0},
            ],
        }
        expected_result = {
            "type": "Soccer",
            "event_date": datetime.strptime("2024-01-01", "%Y-%m-%d").date(),
            "time": "12:30",
            "league": "League",
            "event": "Market",
            "option1": "Option 1",
            "ratio1": 1.5,
            "option2": "Option 2",
            "ratio2": 2.0,
            "option3": None,
            "ratio3": None,
        }
        result = create_bet(market)
        self.assertEqual(result.__dict__, expected_result)

    @patch("awswrangler.s3.to_parquet")
    def test_save_to_s3(self, mock_to_parquet):
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        path = "s3://boaz-winner-api/test-path"
        database = "test-db"
        table = "test-table"
        partition_cols = ["date", "type"]

        save_to_s3(df, path, database, table, partition_cols)

        mock_to_parquet.assert_called_once_with(
            df,
            path=path,
            dataset=True,
            database=database,
            table=table,
            partition_cols=partition_cols,
            mode="append",
        )

    def test_winner_connection(self):
        def fetch_lineChecksum(url: str) -> str:
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                return response.json()["lineChecksum"]
            except:
                raise Exception(
                    f"Failed to fetch lineChecksum from the API: {response.status_code} , {response.content}"
                )

        def fetch_data(url: str, lineChecksum: str) -> dict:
            """Fetch data from the API."""
            try:
                response = requests.get(
                    f"{url}?lineChecksum={lineChecksum}", headers=headers
                )
                return response.json()
            except:
                raise Exception(
                    f"Failed to fetch data from the API: {response.status_code} , {response.content}"
                )

        lineChecksum = fetch_lineChecksum(HASH_CHECKSUM_URL)
        data = fetch_data(API_URL, lineChecksum)
        self.assertIn("hashes", data)
        self.assertIn("markets", data)


if __name__ == "__main__":
    unittest.main()
