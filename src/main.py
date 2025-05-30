import torch
import torch.nn as nn
import random
from transformers import RobertaTokenizer, RobertaModel
from data import get_entries
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import wandb
import torch.nn.functional as F

all_data = get_entries()

vulnerable_data = [i for i in all_data if i.vul == 1]
invulnerable_data = [i for i in all_data if i.vul == 0]
total = len(vulnerable_data) + len(invulnerable_data)

epochs = 5
sample_size = 100
lr = 2e-5
train_samples = vulnerable_data[:sample_size] + invulnerable_data[:sample_size]
random.shuffle(train_samples)

wandb.init(
    project="clinescan",
    config={
        "model": "codebert-base",
        "lr": lr,
        "batch_size": 1,
        "sample_size": sample_size * 2
    }
)

used_ids = set(id(f) for f in train_samples)
eval_samples = [f for f in all_data if id(f) not in used_ids]
eval_samples = random.sample(eval_samples, min(len(train_samples), len(eval_samples)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom weights seem to favour 1 when provided with a lot of data
# weight_for_0 = total / (2 * len(invulnerable_data))
# weight_for_1 = total / (2 * len(vulnerable_data))
# class_weights = torch.tensor([weight_for_0, weight_for_1]).to(device)

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)
model.train()

classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 2)
).to(device)

# Use only with custom weights
# loss_fn = nn.CrossEntropyLoss(weight=class_weights)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(classifier.parameters()), lr=lr
)

global_step = 0
for epoch in range(epochs):
    for i, item in enumerate(train_samples):
        code = "\n".join(item.code)
        label = torch.tensor([item.vul], dtype=torch.long).to(device)

        tokens = tokenizer(code, padding=True, truncation=True, max_length=512, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = model(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        logits = classifier(cls_embedding)
        loss = loss_fn(logits, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Sample {i} | Label: {item.vul} | Loss: {loss.item():.4f}")
        wandb.log({"train/loss": loss.item(), "step": global_step})
        global_step += 1

true_labels = []
predicted_labels = []
all_logits = []
correct = 0

model.eval()
classifier.eval()

results = []
for i, item in enumerate(eval_samples):
    code = "\n".join(item.code)
    true_label = item.vul
    loc = len(item.code)

    tokens = tokenizer(code, padding=True, truncation=True, max_length=512, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = classifier(cls_embedding)

    prob = F.softmax(logits, dim=1)[0, 1].item()
    pred = logits.argmax(dim=1).item()

    true_labels.append(true_label)
    predicted_labels.append(pred)
    all_logits.append(logits)

    results.append({
        "prob": prob,
        "label": true_label,
        "loc": loc
    })
    # predicted = logits.argmax(dim=1).item()
    # predicted_labels.append(predicted)
    # true_labels.append(true_label)

    match = "✅" if pred == true_label else "❌"
    if pred == true_label:
        correct += 1

    print(f"[{match}] Index: {i} | Predicted: {pred} | True: {true_label}")

accuracy = sum([1 for t, p in zip(true_labels, predicted_labels) if t == p]) / len(true_labels)
precision = precision_score(true_labels, predicted_labels, zero_division=0)
recall = recall_score(true_labels, predicted_labels, zero_division=0)
f1 = f1_score(true_labels, predicted_labels, zero_division=0)
cm = confusion_matrix(true_labels, predicted_labels)

print("\n--- Evaluation Metrics ---")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(cm)

# Recall@5, Precision@5
results_sorted = sorted(results, key=lambda x: x["prob"], reverse=True)
top5 = results_sorted[:5]
true_positives_in_top5 = sum(r["label"] for r in top5)
total_vulnerable = sum(r["label"] for r in results)

precision_at_5 = true_positives_in_top5 / 5
recall_at_5 = true_positives_in_top5 / total_vulnerable if total_vulnerable > 0 else 0

# Effort@20% LOC
total_loc = sum(r["loc"] for r in results)
effort_limit = total_loc * 0.2
effort_loc = 0
found_vulnerabilities = 0

for r in results_sorted:
    if effort_loc >= effort_limit:
        break
    effort_loc += r["loc"]
    if r["label"] == 1:
        found_vulnerabilities += 1

effort_20 = found_vulnerabilities / total_vulnerable if total_vulnerable > 0 else 0

print(f"\n--- Top-K Metrics ---")
print(f"Precision@5: {precision_at_5:.2f}")
print(f"Recall@5: {recall_at_5:.2f}")
print(f"Effort@20% LOC: {effort_20:.2f}")

# Log all metrics
wandb.log({
    "eval/accuracy": accuracy,
    "eval/precision": precision,
    "eval/recall": recall,
    "eval/f1": f1,
    "eval/precision@5": precision_at_5,
    "eval/recall@5": recall_at_5,
    "eval/effort@20_LOC": effort_20,
    "eval/confusion_matrix": wandb.plot.confusion_matrix(
        preds=predicted_labels,
        y_true=true_labels,
        class_names=["invulnerable", "vulnerable"]
    )
})