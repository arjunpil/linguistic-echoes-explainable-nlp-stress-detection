# ── Install dependencies ──────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "transformers", "torch", "scikit-learn", 
                "pandas", "numpy", "matplotlib", "seaborn", "-q"])

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, roc_curve)
from sklearn.utils import shuffle

print("GPU available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ── Config ────────────────────────────────────────────────────────
MODEL_NAME  = 'distilbert-base-uncased'
MAX_LEN     = 128
BATCH_SIZE  = 32
EPOCHS      = 3
LR          = 2e-5
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load data ─────────────────────────────────────────────────────
# Update these paths to match where Kaggle puts your dataset
TRAIN_PATH = '/kaggle/input/dreaddit-stress/dreaddit-train.csv'
TEST_PATH  = '/kaggle/input/dreaddit-stress/dreaddit-test.csv'

train = shuffle(pd.read_csv(TRAIN_PATH), random_state=42)
test  = pd.read_csv(TEST_PATH)

print(f"Train: {len(train)} | Test: {len(test)}")
print(train['label'].value_counts())

# ── Dataset class ─────────────────────────────────────────────────
class StressDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts.tolist()
        self.labels    = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Tokenizer & dataloaders ───────────────────────────────────────
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = StressDataset(train['text'], train['label'], tokenizer, MAX_LEN)
test_dataset  = StressDataset(test['text'],  test['label'],  tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

# ── Model ─────────────────────────────────────────────────────────
print(f"\nLoading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = model.to(DEVICE)

# ── Optimizer & scheduler ─────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# ── Training loop ─────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = torch.argmax(outputs.logits, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total

def eval_epoch(model, loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)

            total_loss += outputs.loss.item()
            probs       = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds       = torch.argmax(outputs.logits, dim=1)

            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (total_loss / len(loader), correct / total,
            np.array(all_preds), np.array(all_probs), np.array(all_labels))

# ── Run training ──────────────────────────────────────────────────
print("\nStarting training...")
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
    val_loss, val_acc, val_preds, val_probs, val_labels = eval_epoch(model, test_loader, DEVICE)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

# ── Final evaluation ──────────────────────────────────────────────
print("\n── Final Classification Report ──")
print(classification_report(val_labels, val_preds,
      target_names=['Not Stressed', 'Stressed']))

auc = roc_auc_score(val_labels, val_probs)
print(f"ROC-AUC: {auc:.4f}")

# ── Training curves ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history['train_loss'], label='Train', marker='o')
axes[0].plot(history['val_loss'],   label='Val',   marker='o')
axes[0].set_title('Loss per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history['train_acc'], label='Train', marker='o')
axes[1].plot(history['val_acc'],   label='Val',   marker='o')
axes[1].set_title('Accuracy per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print("Saved: training_curves.png")

# ── Confusion matrix ──────────────────────────────────────────────
cm   = confusion_matrix(val_labels, val_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=['Not Stressed', 'Stressed'])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — DistilBERT', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_distilbert.png', dpi=150)
plt.show()
print("Saved: confusion_matrix_distilbert.png")

# ── ROC curve ─────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(val_labels, val_probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='tomato', lw=2, label=f'DistilBERT (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — DistilBERT')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve_distilbert.png', dpi=150)
plt.show()
print("Saved: roc_curve_distilbert.png")

# ── Save model ────────────────────────────────────────────────────
model.save_pretrained('/kaggle/working/distilbert_stress')
tokenizer.save_pretrained('/kaggle/working/distilbert_stress')
print("\n✅ Model saved to /kaggle/working/distilbert_stress")