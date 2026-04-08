import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer

from src.model import Net
from src.preprocess import load_csv_dataset


TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "log_model.pt"
META_PATH = ARTIFACTS_DIR / "meta.json"


def evaluate(model: Net, x_test: torch.Tensor, y_test: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(x_test)
        binary_preds = (preds > 0.5).float()
        accuracy = (binary_preds.eq(y_test).sum().item()) / len(y_test)
    return accuracy


def train_model(epochs: int = 200) -> Tuple[Net, SentenceTransformer]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    train_logs, train_labels = load_csv_dataset(TRAIN_PATH)
    test_logs, test_labels = load_csv_dataset(TEST_PATH)

    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    x_train = torch.tensor(embedder.encode(train_logs), dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)

    x_test = torch.tensor(embedder.encode(test_logs), dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

    model = Net(x_train.shape[1])

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()

        output = model(x_train)
        loss = criterion(output, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    accuracy = evaluate(model, x_test, y_test)
    print(f"\nTest accuracy: {accuracy:.2%}")

    torch.save(model.state_dict(), MODEL_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": "all-MiniLM-L6-v2",
                "threshold": 0.5,
                "input_size": x_train.shape[1],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {META_PATH}")

    return model, embedder