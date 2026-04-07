import torch
import torch.nn as nn
import torch.optim as optim

# ===== 1. Дані =====
logs = [
    "login failure for user admin",
    "ssh brute force detected",
    "multiple login attempts",
    "normal connection established",
    "user logged in successfully",
    "system running normally"
]

labels = [1, 1, 1, 0, 0, 0]  # 1 = attack, 0 = normal


# ===== 2. Токенізація =====
def tokenize(text):
    return text.lower().split()


vocab = {}
idx = 0

for log in logs:
    for word in tokenize(log):
        if word not in vocab:
            vocab[word] = idx
            idx += 1


def vectorize(text):
    vec = [0] * (len(vocab) + 1)  # +1 для unknown

    for word in tokenize(text):
        if word in vocab:
            vec[vocab[word]] += 1
        else:
            vec[-1] += 1  # unknown слова

    return vec


X = torch.tensor([vectorize(log) for log in logs], dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)


# ===== 3. Модель =====
class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


model = Net(len(vocab) + 1)


# ===== 4. Тренування =====
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    output = model(X)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# ===== 5. Тест =====
test_log = "admin hacked"
test_vec = torch.tensor(vectorize(test_log), dtype=torch.float32)

prediction = model(test_vec)
print("\nTest:", test_log)

if prediction.item() > 0.5:
    print("Prediction: ATTACK")
else:
    print("Prediction: NORMAL")