import pandas as pd
import numpy as np
from nlpimdbmoviereviews.constants import (
    DATA_COLUMNS,
    ALL_GENRES,
    GENRE_IDS,
    GENRE_NAME,
    GENRE_ID,
)


class BaseValidator:
    def validate(
        self, overview_data: pd.DataFrame, genre_data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Subclasses should implement this method")


class ColumnValidator(BaseValidator):
    def validate(
        self, overview_data: pd.DataFrame, genre_data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        missing_columns = [
            col
            for col in DATA_COLUMNS
            if col not in overview_data.columns and col not in genre_data.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        missing_columns = [
            name for name in ALL_GENRES if name not in genre_data[GENRE_NAME].values
        ]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        extra_columns = [
            name for name in genre_data[GENRE_NAME].values if name not in ALL_GENRES
        ]
        if extra_columns:
            print(f"Extra columns: {', '.join(extra_columns)}")
        return np.array([True] * len(overview_data)), ~genre_data[GENRE_NAME].isin(
            extra_columns
        )


class GenreIdValidator(BaseValidator):
    def validate(
        self, overview_data: pd.DataFrame, genre_data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        valid_rows = np.ones((len(overview_data)), dtype=bool)
        for i, ids in enumerate(overview_data[GENRE_IDS]):
            genre_ids = ids.strip("[]").replace(" ", "")
            if genre_ids == "":
                print(f"Empty genre IDs in column {i}")
                valid_rows[i] = False
            else:
                unknown_genre_ids = [
                    genre_id
                    for genre_id in genre_ids.split(",")
                    if int(genre_id) not in genre_data[GENRE_ID].values
                ]
                if unknown_genre_ids:
                    print(
                        f"Unknown genre ID in column {i}: {', '.join(unknown_genre_ids)}"
                    )
                    valid_rows[i] = False
        return valid_rows, np.array([True] * len(genre_data))
