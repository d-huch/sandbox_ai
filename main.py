import re
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer


# =========================
# 1. Нормалізація логів
# =========================
def normalize_log(text: str) -> str:
    text = text.lower().strip()

    # IP адреси
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<IP>", text)

    # дати/час типу 09:15:03 або 2026-04-01
    text = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "<TIME>", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "<DATE>", text)

    # числа
    text = re.sub(r"\b\d+\b", "<NUM>", text)

    # зайві пробіли
    text = re.sub(r"\s+", " ", text)

    return text


# =========================
# 2. Дані
# 1 = attack, 0 = normal
# =========================
raw_logs = [
    "login failure for user admin from 45.205.1.5",
    "ssh brute force detected from 112.46.215.69",
    "multiple login attempts for root from 109.231.155.82",
    "unauthorized access attempt from 171.231.178.54",
    "port scan detected from 10.10.10.5",
    "failed api authentication from 45.205.1.110",
    "normal connection established",
    "user logged in successfully",
    "system running normally",
    "configuration saved successfully",
    "backup completed without errors",
    "service restarted normally",
]

labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

logs = [normalize_log(x) for x in raw_logs]

# train/test split вручну
train_logs = logs[:8]
train_labels = labels[:8]

test_logs = logs[8:]
test_labels = labels[8:]


# =========================
# 3. Embeddings
# =========================
device = "cpu"
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

X_train = torch.tensor(embedder.encode(train_logs), dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(embedder.encode(test_logs), dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)


# =========================
# 4. Модель
# =========================
class Net(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


model = Net(X_train.shape[1])


# =========================
# 5. Тренування
# =========================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 200

for epoch in range(epochs):
    model.train()

    output = model(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 25 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")


# =========================
# 6. Оцінка на test
# =========================
model.eval()

with torch.no_grad():
    preds = model(X_test)
    binary_preds = (preds > 0.5).float()
    accuracy = (binary_preds.eq(y_test).sum().item()) / len(y_test)

print(f"\nTest accuracy: {accuracy:.2%}")

print("\nTest samples:")
for log, pred, true_label in zip(test_logs, preds, test_labels):
    score = pred.item()
    predicted_label = 1 if score > 0.5 else 0
    predicted_text = "ATTACK" if predicted_label == 1 else "NORMAL"
    true_text = "ATTACK" if true_label == 1 else "NORMAL"

    print(f"- Log: {log}")
    print(f"  Score: {score:.4f}")
    print(f"  Predicted: {predicted_text} | True: {true_text}")


# =========================
# 7. Збереження
# =========================
save_dir = Path("artifacts")
save_dir.mkdir(exist_ok=True)

torch.save(model.state_dict(), save_dir / "log_model.pt")

with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "model_name": "all-MiniLM-L6-v2",
            "threshold": 0.5,
            "note": "Simple log classifier using sentence embeddings + MLP",
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\nModel saved to artifacts/log_model.pt")
print("Metadata saved to artifacts/meta.json")


# =========================
# 8. Інтерактивний режим
# =========================
print("\nInteractive mode. Type 'exit' to quit.")

while True:
    user_input = input("\nEnter log: ").strip()

    if user_input.lower() == "exit":
        print("Bye.")
        break

    normalized = normalize_log(user_input)
    vec = torch.tensor(embedder.encode([normalized]), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred = model(vec)
        score = pred.item()

    print(f"Normalized: {normalized}")
    print(f"Score: {score:.4f}")
    print("Prediction:", "ATTACK" if score > 0.5 else "NORMAL")