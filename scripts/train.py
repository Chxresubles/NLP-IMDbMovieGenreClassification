import json
import torch
import pickle
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from nlpimdbmoviereviews.models import RoBERTa
from nlpimdbmoviereviews.trainer import ModelTrainer
from nlpimdbmoviereviews.dataloader import DataLoader
from nlpimdbmoviereviews.dataset import MovieGenreDataset
from nlpimdbmoviereviews.constants import OVERVIEW, SEED
from nlpimdbmoviereviews.validators import (
    ColumnValidator,
    GenreIdValidator,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train NLP Movie Genre classification model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to train the model",
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
        "--train-split",
        type=float,
        default=0.8,
        help="Ratio of training data (default: 0.8)",
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
    features, labels = dataloader.get_data(drop_hard_genres=args.drop_hard_genres)

    X = features[OVERVIEW].to_list()
    y = labels.values.astype(int)

    # Create features and labels for training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - args.train_split, random_state=SEED
    )

    train_dataset = MovieGenreDataset(X_train, y_train)
    print(f"Train dataset contains {len(train_dataset)} movies")

    test_dataset = MovieGenreDataset(X_test, y_test)
    print(f"Test dataset contains {len(test_dataset)} movies")

    model = RoBERTa(y.shape[1])

    trainer = ModelTrainer(model)

    train_metrics = trainer.train(train_dataset, n_epochs=args.epochs)
    print(f"Model train metrics = {train_metrics}")

    test_metrics = trainer.evaluate(test_dataset)
    print(f"Model test metrics = {test_metrics}")

    out_path = Path(args.output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "model.pkl", "wb") as f:
        pickle.dump(model.to("cpu"), f)

    with open(out_path / "train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)

    with open(out_path / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
