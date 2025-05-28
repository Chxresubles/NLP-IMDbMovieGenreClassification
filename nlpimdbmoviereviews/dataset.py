import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MovieGenreDataset(Dataset):
    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame) -> None:
        self.features = features
        self.labels = labels
        assert len(features) == len(
            labels
        ), "Features and labels must have the same length !"

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray]:
        return self.features[idx], self.labels[idx]
