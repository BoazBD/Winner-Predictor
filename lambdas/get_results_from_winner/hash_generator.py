import hashlib
import pandas as pd


class HashGenerator:
    def create_hash(
        self, type: str, date: str, league: str, team1: str, team2: str
    ) -> str:
        data_to_hash = f"{type}_{date}_{league}_{team1}_{team2}"
        return hashlib.sha1(data_to_hash.encode()).hexdigest()[:16]

    def generate_unique_id(self, row):
        if not pd.isnull(row["option3"]):
            second_team = row["option3"]
        else:
            second_team = row["option2"]

        return self.create_hash(
            row["type"], row["event_date"], row["league"], row["option1"], second_team
        )

    def generate_result_id(self, row: pd.Series) -> str:
        first_team = self.clean_team_name(row["option1"])
        # Don't use ternary operator here, as it will always evaluate both sides
        if not pd.isnull(row["option3"]):
            second_team = self.clean_team_name(row["option3"])
        else:
            second_team = self.clean_team_name(row["option2"])

        return self.create_hash(
            row["type"], row["event_date"], row["league"], first_team, second_team
        )

    def clean_team_name(self, team_name: str) -> str:
        """
        This function cleans a team name by removing any parentheses and numbers at the end.
        """
        # Find the last index of an opening parenthesis
        opening_bracket_index = team_name.rfind("(")
        # If an opening parenthesis is found, remove everything after it (including the parenthesis)
        if opening_bracket_index != -1:
            return team_name[:opening_bracket_index].strip()
        else:
            return team_name 