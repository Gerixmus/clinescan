import torch
import torch.nn as nn
import random
from transformers import RobertaTokenizer, RobertaModel
import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_data, train_samples, used_indices, remaining_indices = data.get_entries(100)

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)
model.eval()

classifier = nn.Sequential(
    nn.Linear(768, 2)
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)

for i, item in enumerate(train_samples):
    code = item["code"]
    label = torch.tensor([item["vul"]]).to(device)

    tokens = tokenizer(code, padding=True, truncation=True, max_length=512, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens)
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    logits = classifier(cls_embedding)
    loss = loss_fn(logits, label)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Sample {i} | Label: {item['vul']} | Loss: {loss.item():.4f}")

eval_indices = random.sample(remaining_indices, min(len(train_samples), len(remaining_indices)))

correct = 0

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

    match = "✅" if predicted == true_label else "❌"
    if predicted == true_label:
        correct += 1

    print(f"[{match}] Index: {i} | Predicted: {predicted} | True: {true_label}")

print(f"\nAccuracy: {correct}/{len(train_samples)} = {100 * correct / len(train_samples):.2f}%")
