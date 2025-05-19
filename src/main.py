import torch
import torch.nn as nn
import random
from transformers import RobertaTokenizer, RobertaModel
import data
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

#TODO move to data
num_vul = 8794
num_invul = 177736
total = num_vul + num_invul

weight_for_0 = total / (2 * num_invul)
weight_for_1 = total / (2 * num_vul)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_weights = torch.tensor([weight_for_0, weight_for_1]).to(device)

all_data, train_samples, used_indices, remaining_indices = data.get_entries(1000)

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

loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(classifier.parameters()), lr=2e-5
)

for i, item in enumerate(train_samples):
    code = item["code"]
    label = torch.tensor([item["vul"]]).to(device)

    tokens = tokenizer(code, padding=True, truncation=True, max_length=512, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}

    outputs = model(**tokens)
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    logits = classifier(cls_embedding)
    loss = loss_fn(logits, label)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Sample {i} | Label: {item['vul']} | Loss: {loss.item():.4f}")

eval_indices = random.sample(remaining_indices, min(len(train_samples), len(remaining_indices)))
true_labels = []
predicted_labels = []
correct = 0

model.eval()
classifier.eval()

for i in eval_indices:
    code = all_data[i]["code"]
    true_label = all_data[i]["vul"]

    tokens = tokenizer(code, padding=True, truncation=True, max_length=512, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = classifier(cls_embedding)

    predicted = logits.argmax(dim=1).item()
    predicted_labels.append(predicted)
    true_labels.append(true_label)

    match = "✅" if predicted == true_label else "❌"
    if predicted == true_label:
        correct += 1

    print(f"[{match}] Index: {i} | Predicted: {predicted} | True: {true_label}")

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
