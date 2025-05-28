import json
import torch
import pickle
import argparse
import numpy as np
from pathlib import Path
from nlpimdbmoviereviews.trainer import ModelTrainer
from nlpimdbmoviereviews.dataloader import DataLoader
from nlpimdbmoviereviews.dataset import MovieGenreDataset
from nlpimdbmoviereviews.constants import OVERVIEW, SEED, HARD_CLASSES
from nlpimdbmoviereviews.validators import (
    ColumnValidator,
    GenreIdValidator,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate NLP Movie Genre classification model"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to input data folder",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./output",
        help="Path to output folder",
    )
    parser.add_argument(
        "--drop-hard-genres",
        action="store_true",
        help="Drop movie genres that are hardly predictable with the only synopsis.",
    )
    args = parser.parse_args()

    np.random.seed(seed=SEED)
    torch.manual_seed(seed=SEED)

    dataloader = DataLoader(
        Path(args.data_path) / "movies_overview.csv",
        Path(args.data_path) / "movies_genres.csv",
        [ColumnValidator(), GenreIdValidator()],
    )
    features, labels = dataloader.get_data()

    X = features[OVERVIEW].to_list()
    y = labels.values.astype(int)

    out_path = Path(args.output_path)
    with open(out_path / "model.pkl", "rb") as f:
        model = pickle.load(f)

    validation_dataset = MovieGenreDataset(X, y)
    print(f"Validation dataset contains {len(validation_dataset)} movies")

    trainer = ModelTrainer(model)

    validation_metrics = trainer.evaluate(validation_dataset)
    print(f"Model validation metrics = {validation_metrics}")

    with open(out_path / "validation_metrics.json", "w") as f:
        json.dump(validation_metrics, f, indent=2)
