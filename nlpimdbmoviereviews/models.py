import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, RobertaForSequenceClassification


class RoBERTa(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.vectorizer = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-emotion"
        )
        self.roberta_classifier = RobertaForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-emotion",
            problem_type="multi_label_classification",
        )
        self.roberta_classifier.classifier.out_proj = nn.Linear(
            in_features=self.roberta_classifier.config.hidden_size,
            out_features=output_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Get word representation from tokenizer
        out = self.vectorizer(
            x,
            padding="longest",
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Transfer tensors to the correct device
        device = next(self.parameters()).device
        out["input_ids"] = out["input_ids"].to(device)
        out["attention_mask"] = out["attention_mask"].to(device)

        # Model inference
        out = self.roberta_classifier(
            input_ids=out["input_ids"], attention_mask=out["attention_mask"]
        ).logits
        if self.training:
            return out
        return torch.sigmoid(out)
