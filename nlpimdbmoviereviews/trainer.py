from typing import Optional
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from nlpimdbmoviereviews.constants import BATCH_SIZE
from sklearn.metrics import hamming_loss, classification_report, accuracy_score


class ModelTrainer:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading tensors on {self.device.type}")

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        n_epochs: Optional[int] = 20,
        lr: Optional[float] = 1e-4,
    ) -> dict:

        # Create DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
        )

        criterion = nn.BCEWithLogitsLoss()
        optim = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)

        self.model.train()
        self.model.to(device=self.device)
        for i in range(n_epochs):
            tot_loss = 0.0
            y_pred = []
            y_true = []
            for data, target in tqdm(dataloader, leave=False):
                y_true.append(target)
                target = target.float().to(self.device)

                output = self.model(data)

                loss = criterion(output, target)
                loss.backward()
                optim.step()
                optim.zero_grad()

                tot_loss += loss.item() * len(data)
                y_pred.append(torch.sigmoid(output).round().cpu().detach().numpy())

            scheduler.step()

            y_pred = np.vstack(y_pred)
            y_true = np.vstack(y_true)
            with torch.no_grad():
                # Compute metrics
                hamming_acc = 1 - hamming_loss(y_true, y_pred)
                subset_accuracy = accuracy_score(y_true, y_pred)
                report = classification_report(
                    y_true, y_pred, zero_division=0, output_dict=True
                )

            # Average loss across batches
            tot_loss /= len(dataset)

            print(
                f"epoch {i + 1}/{n_epochs}: loss={tot_loss} Hamming accuracy={hamming_acc} Subset accuracy={subset_accuracy} Micro average F1-score={report['micro avg']['f1-score']} Macro average F1-score={report['macro avg']['f1-score']}"
            )

        return self.evaluate(dataset)

    def evaluate(self, dataset: torch.utils.data.Dataset) -> dict:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
        )

        self.model.eval()
        self.model.to(device=self.device)
        with torch.no_grad():
            y_pred = []
            y_true = []
            for data, target in tqdm(dataloader, leave=False):
                y_true.append(target)
                target = target.float().to(self.device)
                output = self.model(data)
                y_pred.append(output.round().cpu().numpy())
            y_pred = np.vstack(y_pred)
            y_true = np.vstack(y_true)

            # Compute metrics
            hamming_acc = 1 - hamming_loss(y_true, y_pred)
            subset_accuracy = accuracy_score(y_true, y_pred)
            report = classification_report(
                y_true, y_pred, zero_division=0, output_dict=True
            )

        return {
            "Hamming accuracy": hamming_acc,
            "Subset accuracy": subset_accuracy,
            "Micro average F1-score": report["micro avg"]["f1-score"],
            "Macro average F1-score": report["macro avg"]["f1-score"],
        }
