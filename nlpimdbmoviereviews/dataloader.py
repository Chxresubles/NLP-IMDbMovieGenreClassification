from typing import Optional
import pandas as pd
from nlpimdbmoviereviews.validators import BaseValidator
from nlpimdbmoviereviews.constants import (
    GENRE_IDS,
    GENRE_ID,
    GENRE_NAME,
    OVERVIEW,
    TITLE,
    PREDICTABLE_GENRES,
    NARROW_PREDICTABLE_GENRES,
)


class DataLoader:
    def __init__(
        self,
        overview_file_path: str,
        genre_file_path: str,
        validators: Optional[list[BaseValidator]] = [],
    ) -> None:
        self.overview_file_path = overview_file_path
        self.genre_file_path = genre_file_path
        self.validators = validators
        self.overview_data = None
        self.genre_data = None
        self.features = None
        self.labels = None

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        overview_data = pd.read_csv(self.overview_file_path)
        genre_data = pd.read_csv(self.genre_file_path)
        return overview_data, genre_data

    def _build_data(
        self,
        overview_data: pd.DataFrame,
        genre_data: pd.DataFrame,
        drop_hard_genres: Optional[bool] = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Create one column per genre
        data = overview_data.copy()
        genre_list = (
            NARROW_PREDICTABLE_GENRES if drop_hard_genres else PREDICTABLE_GENRES
        )
        for _, row in genre_data.iterrows():
            if row[GENRE_NAME] in genre_list:
                col_id = str(row[GENRE_ID])
                data[row[GENRE_NAME]] = data[GENRE_IDS].apply(
                    lambda x: col_id in x.strip("[]").replace(" ", "").split(",")
                )
        data.drop(columns=[GENRE_IDS], inplace=True)

        # Combine rows with duplicate descriptions
        overviews = data[OVERVIEW]
        if overviews.duplicated().sum() > 0:
            duplicate_descriptions = data[overviews.duplicated(keep="first")][OVERVIEW]
            for descr in duplicate_descriptions:
                data.loc[overviews == descr, TITLE] = data.loc[
                    overviews == descr, TITLE
                ].iloc[0]
                for col in data.columns:
                    if col != TITLE and col != OVERVIEW:
                        data.loc[overviews == descr, col] = data.loc[
                            overviews == descr, col
                        ].any()
            data.drop_duplicates(inplace=True)

        # Drop column with single value
        single_value_columns = data.nunique() == 1
        data.drop(
            columns=single_value_columns[single_value_columns].index, inplace=True
        )

        # Drop rows with no associated classes
        empty_genre_rows = data.drop([TITLE, OVERVIEW], axis=1).any(axis=1)
        data = data.loc[empty_genre_rows, :]

        # Drop duplicates
        data.drop_duplicates(inplace=True)

        # Separate features and labels
        features = data[[TITLE, OVERVIEW]]
        labels = data[[genre for genre in genre_list if genre in data.columns.values]]
        return features, labels

    def _validate_data(
        self, overview_data: pd.DataFrame, genre_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        for validator in self.validators:
            overview_indexes, genre_indexes = validator.validate(
                overview_data, genre_data
            )
            overview_data = overview_data[overview_indexes]
            genre_data = genre_data[genre_indexes]
        return overview_data, genre_data

    def get_data(
        self, drop_hard_genres: Optional[bool] = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.overview_data, self.genre_data = self._load_data()

        self.overview_data = self.overview_data.drop_duplicates()
        self.overview_data = self.overview_data.dropna()

        self.genre_data = self.genre_data.drop_duplicates()
        self.genre_data = self.genre_data.dropna()

        self.overview_data, self.genre_data = self._validate_data(
            self.overview_data, self.genre_data
        )

        self.features, self.labels = self._build_data(
            self.overview_data, self.genre_data, drop_hard_genres=drop_hard_genres
        )

        return self.features, self.labels
