from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim

# модель embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ===== дані =====
logs = [
    "login failure for user admin",
    "ssh brute force detected",
    "multiple login attempts",
    "normal connection established",
    "user logged in successfully",
    "system running normally"
]

labels = [1, 1, 1, 0, 0, 0]

# ===== embeddings =====
X = torch.tensor(embedder.encode(logs))
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)


# ===== 3. Модель =====
class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

# ===== 4. Тренування =====
model = Net(X.shape[1])

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    output = model(X)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ===== 5. Тест =====
test_log = "Denisky Vzlomaly"
test_vec = torch.tensor(embedder.encode([test_log]))

pred = model(test_vec)

print(test_log)
print("ATTACK" if pred.item() > 0.5 else "NORMAL")