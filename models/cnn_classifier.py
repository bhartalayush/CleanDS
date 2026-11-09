"""
cnn_classifier.py
A Character-Level 1D Convolutional Neural Network (1D-CNN) in PyTorch.
Classifies column data roles by analyzing character-level n-grams and layout patterns.

Classes:
0: Numeric (e.g. '123.45', '98120')
1: Date_Time (e.g. '2024-11-09', '03/12/2021', '2023-01-01 10:30')
2: Currency (e.g. '$1,299.99', '€45.50', '₹50,000')
3: Identifier (e.g. 'USR-9021', 'A9B-1029', 'TX99042')
4: Categorical_String (e.g. 'Active', 'Pending', 'Male', 'California')
5: Free_Text (e.g. 'Long customer feedback notes about order delay')
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

CLASSES = [
    "Numeric",
    "Date_Time",
    "Currency",
    "Identifier",
    "Categorical_String",
    "Free_Text"
]

class Char1DCNN(nn.Module):
    def __init__(self, vocab_size=128, embed_dim=32, num_classes=len(CLASSES)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Hidden Level 1: Local n-gram feature extraction
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        # Hidden Level 2: High-order structure composition
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Hidden Level 3: Classification head
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.25)
        self.out = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, seq_len)
        emb = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq_len)
        
        h1 = self.pool1(self.relu(self.bn1(self.conv1(emb))))
        h2 = self.global_pool(self.relu(self.bn2(self.conv2(h1)))).squeeze(-1)
        
        feat = self.dropout(self.relu(self.fc1(h2)))
        return self.out(feat)


def encode_string(text: str, max_len: int = 64) -> list:
    """Encodes ASCII characters into integer indices bounded by 127."""
    clean_text = str(text)[:max_len]
    encoded = [ord(c) if ord(c) < 128 else 0 for c in clean_text]
    # Pad with zeros
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    return encoded


class ColumnClassifier:
    def __init__(self, weights_path="models/weights/cnn_column_classifier.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Char1DCNN().to(self.device)
        self.weights_path = weights_path
        
        if os.path.exists(self.weights_path):
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device, weights_only=True))
            self.model.eval()
        else:
            self._train_and_save()

    def _train_and_save(self):
        """Generates synthetic training data and trains the 1D-CNN in ~2 seconds."""
        synthetic_samples = []
        labels = []

        # 0: Numeric (clean numbers, NO currency symbols)
        for _ in range(300):
            val = str(random.choice([
                random.randint(-1000, 100000),
                round(random.uniform(-500.0, 50000.0), 2),
                f"{random.randint(10, 99)},{random.randint(100, 999)}.{random.randint(10, 99)}"
            ]))
            synthetic_samples.append(val)
            labels.append(0)

        # 1: Date_Time
        for _ in range(300):
            val = random.choice([
                f"{random.randint(2015, 2026)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                f"{random.randint(1, 12):02d}/{random.randint(1, 28):02d}/{random.randint(2015, 2026)}",
                f"{random.randint(2020, 2026)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00"
            ])
            synthetic_samples.append(val)
            labels.append(1)

        # 2: Currency (always includes $, EUR, USD, INR, etc.)
        for _ in range(300):
            symbol = random.choice(["$", "€", "£", "₹", "USD ", "EUR "])
            val = f"{symbol}{random.randint(10, 9999):,}.{random.randint(10, 99):02d}"
            synthetic_samples.append(val)
            labels.append(2)

        # 3: Identifier
        for _ in range(250):
            prefix = random.choice(["ID", "USR", "EMP", "TX", "ORD", "CUST", "SKU"])
            val = f"{prefix}-{random.randint(1000, 99999)}"
            synthetic_samples.append(val)
            labels.append(3)

        # 4: Categorical_String
        categories = ["Active", "Inactive", "Pending", "Approved", "Male", "Female", "High", "Medium", "Low", "California", "Texas", "New York"]
        for _ in range(250):
            synthetic_samples.append(random.choice(categories))
            labels.append(4)

        # 5: Free_Text
        texts = [
            "Customer requested delivery before noon due to office closure",
            "Follow-up required with supplier regarding shipment delay",
            "User experienced intermittent login errors during authentication",
            "General comment regarding invoice discrepancy and tax calculations"
        ]
        for _ in range(250):
            synthetic_samples.append(random.choice(texts))
            labels.append(5)

        X = torch.tensor([encode_string(s) for s in synthetic_samples], dtype=torch.long).to(self.device)
        y = torch.tensor(labels, dtype=torch.long).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.003)

        self.model.train()
        for epoch in range(15):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        self.model.eval()
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.weights_path)

    def predict_column(self, sample_values: list) -> dict:
        """
        Runs batch inference across sample values from a column.
        Averages softmax probabilities to provide a robust column prediction.
        """
        valid_samples = [str(v).strip() for v in sample_values if str(v).strip() and str(v).lower() != 'nan']
        if not valid_samples:
            return {"role": "Unknown", "confidence": 0.0}

        # Subsample up to 25 items for fast inference
        valid_samples = valid_samples[:25]
        encoded = [encode_string(s) for s in valid_samples]
        x_tensor = torch.tensor(encoded, dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=-1)
            avg_probs = probs.mean(dim=0).cpu().numpy()

        best_idx = int(avg_probs.argmax())
        confidence = float(avg_probs[best_idx])

        return {
            "role": CLASSES[best_idx],
            "confidence": round(confidence * 100, 1),
            "class_probabilities": {CLASSES[i]: round(float(avg_probs[i]) * 100, 1) for i in range(len(CLASSES))}
        }
