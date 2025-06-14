import torch
import torch.nn as nn
import random
from transformers import RobertaModel, RobertaTokenizerFast
from data import get_entries
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import wandb
import torch.nn.functional as F
from torch import amp
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

scaler = amp.GradScaler('cuda')

all_data = get_entries()
labels = [entry.vul for entry in all_data]

epochs = 3
sample_size = 0.03
lr = 1e-5
seed = 42
batch_size = 1

set_seed(seed)

wandb.init(
    project="clinescan-eval",
    name=f"{epochs}x-{sample_size*100}%",
    config={
        "model": "codebert-base",
        "lr": lr,
        "batch_size": batch_size,
        "sample_size": sample_size
    }
)

vulnerable = [s for s in all_data if s.vul == 1]
non_vulnerable = [s for s in all_data if s.vul == 0]

train_samples, eval_samples = train_test_split(
    all_data,
    test_size = sample_size * 0.2,
    random_state = seed,
    shuffle = True,
    stratify = labels
)

vul_train = [x for x in train_samples if x.vul == 1]
nonvul_train = [x for x in train_samples if x.vul == 0]

balanced_train_size = int(len(all_data) * sample_size * 0.8)

half_size = balanced_train_size // 2

vul_train_balanced = random.sample(vul_train, min(len(vul_train), half_size))
nonvul_train_balanced = random.sample(nonvul_train, min(len(nonvul_train), half_size))

balanced_train_samples = vul_train_balanced + nonvul_train_balanced
random.shuffle(balanced_train_samples)

print(f"Train samples: {len(balanced_train_samples)}")
print(f"Eval samples: {len(eval_samples)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

model = RobertaModel.from_pretrained("microsoft/codebert-base", attn_implementation="eager").to(device)
model.train()

classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 1)
).to(device)

def get_smoothing_loss_fn(epoch):
    smoothing = max(0.05, 0.1 * (0.9 ** epoch))
    return nn.CrossEntropyLoss(label_smoothing=smoothing)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(classifier.parameters()), lr=lr
)

for epoch in range(epochs):
    loss_fn = get_smoothing_loss_fn(epoch)
    for i, item in enumerate(balanced_train_samples):
        code = "\n".join(item.code)
        label = torch.tensor([item.vul], dtype=torch.float32).to(device) 

        tokens = tokenizer(
            code, padding=True, truncation=True, max_length=512,
            return_tensors="pt", return_offsets_mapping=True
        )
        offset_mapping = tokens.pop("offset_mapping")[0].tolist()
        tokens = {k: v.to(device) for k, v in tokens.items()}

        optimizer.zero_grad()

        with amp.autocast(device_type='cuda'):
            outputs = model(**tokens, output_attentions=True)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            logits = classifier(cls_embedding).view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, label)

        if label.item() == 1 and item.flaw_line_no:
            attn = torch.stack(outputs.attentions[4:8])[:, 0, :, 0, :]  # [layers, heads, seq_len]
            mean_attn = attn.mean(dim=(0, 1))  # [seq_len]

            lines = code.splitlines()
            token_to_line = {}
            current_line = 0
            char_pointer = 0
            for i_tok, (start, end) in enumerate(offset_mapping):
                while current_line < len(lines) and start >= char_pointer + len(lines[current_line]):
                    char_pointer += len(lines[current_line]) + 1
                    current_line += 1
                token_to_line[i_tok] = current_line

            valid_indices = [i for i in range(len(mean_attn)) if token_to_line.get(i, -1) >= 0]
            mean_attn = mean_attn[valid_indices]
            match_mask = torch.tensor([
                1.0 if token_to_line[i] + 1 in item.flaw_line_no else 0.0
                for i in valid_indices
            ], device=mean_attn.device)

            mean_attn = torch.clamp(mean_attn, min=1e-6, max=1-1e-6)
            attn_probs = F.softmax(mean_attn, dim=0)

            if match_mask.sum() > 0:
                match_mask = match_mask / match_mask.sum()
                attention_loss = F.kl_div(attn_probs.log(), match_mask, reduction="batchmean")
                loss = loss + 0.2 * attention_loss
            else:
                loss = loss * 1.2


        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()

        print(f"Epoch {epoch} Sample {i} | Loss: {loss.item():.4f}")
        wandb.log({"train/loss": loss.item(), "step": epoch * len(train_samples) + i})

model.eval()
classifier.eval()

true_labels = []
predicted_labels = []

recall_at_5, precision_at_5, effort_20 = [], [], []

with torch.no_grad():
    for item in tqdm(eval_samples):
        code = "\n".join(item.code)
        flaw_lines = item.flaw_line_no
        if not flaw_lines:
            continue

        tokens = tokenizer(
            code, return_offsets_mapping=True,
            padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        )
        offset_mapping = tokens.pop("offset_mapping")[0].tolist()
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = model(**tokens, output_attentions=True)
        attn_layers = outputs.attentions[4:8]
        # weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device).view(-1, 1, 1, 1, 1)
        # attn = torch.stack(attn_layers)[:, 0, :, 0, :]  # shape: [4, heads, seq_len]
        # mean_attn = (attn * weights).sum(dim=0).mean(dim=0)  # weighted avg, then mean over heads

        lines = code.splitlines()
        token_to_line = {}
        current_line = 0
        char_pointer = 0
        for i_tok, (start, end) in enumerate(offset_mapping):
            while current_line < len(lines) and start >= char_pointer + len(lines[current_line]):
                char_pointer += len(lines[current_line]) + 1
                current_line += 1
            token_to_line[i_tok] = current_line

        line_scores = {}

        weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device).view(-1, 1, 1)  # [4,1,1]
        attn = torch.stack(attn_layers)[:, 0, :, 0, :]  # [4, heads, seq_len]
        weighted_attn = attn * weights  # broadcasts properly
        mean_attn = weighted_attn.sum(dim=0).mean(dim=0)  # [seq_len]

        for idx, score in enumerate(mean_attn):
            line = token_to_line.get(idx, None)
            if line is not None:
                line_scores[line] = line_scores.get(line, 0.0) + score.item()

        if line_scores:
            scores = torch.tensor(list(line_scores.values()))
            scores = F.softmax(scores, dim=0)
            line_scores = {k: scores[i].item() for i, k in enumerate(line_scores.keys())}

            sorted_lines = sorted(line_scores.items(), key=lambda x: x[1], reverse=True)

            top5 = [lineno + 1 for lineno, _ in sorted_lines[:5]]
            topn = max(1, int(0.2 * len(lines)))
            top20p = [lineno + 1 for lineno, _ in sorted_lines[:topn]]

            hit = len(set(top5) & set(flaw_lines)) > 0
            recall_at_5.append(1.0 if hit else 0.0)

            precision_hits = sum(1 for l in top5 if l in flaw_lines)
            precision_at_5.append(precision_hits / 5)

            effort_hits = sum(1 for l in flaw_lines if l in top20p)
            effort_20.append(effort_hits / len(flaw_lines))

with torch.no_grad():
    for item in eval_samples:
        code = "\n".join(item.code)
        label = item.vul
        tokens = tokenizer(code, padding=True, truncation=True, max_length=512, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        outputs = model(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = classifier(cls_embedding).squeeze()
        pred = (logits > 0).long().item()

        true_labels.append(label)
        predicted_labels.append(pred)

# === [PRINT METRICS] ===
if recall_at_5:
    recall5 = sum(recall_at_5) / len(recall_at_5)
    precision5 = sum(precision_at_5) / len(precision_at_5)
    effort20 = sum(effort_20) / len(effort_20)

    print("\n=== Line-Level Metrics ===")
    print(f"Recall@5        : {recall5:.4f}")
    print(f"Precision@5     : {precision5:.4f}")
    print(f"Effort@20% LOC  : {effort20:.4f}")

    wandb.log({
    "line_level/recall@5": recall5,
    "line_level/precision@5": precision5,
    "line_level/effort@20%_LOC": effort20
    })

accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels)
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

wandb.log({
    "eval/accuracy": accuracy,
    "eval/precision": precision,
    "eval/recall": recall,
    "eval/f1": f1,
    "eval/confusion_matrix": wandb.plot.confusion_matrix(
        preds=predicted_labels,
        y_true=true_labels,
        class_names=["invulnerable", "vulnerable"]
    )
})