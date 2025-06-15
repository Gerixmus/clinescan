import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm
from function import Function
from config import Config
from logger import WandbLogger


def evaluate(
        model, 
        eval_samples: list[Function],
        config: Config,
        tokenizer, 
        classifier,
        logger: WandbLogger
    ):

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
            tokens = {k: v.to(config.device) for k, v in tokens.items()}

            outputs = model(**tokens, output_attentions=True)
            attn_layers = outputs.attentions[4:8]

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

            weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=config.device).view(-1, 1, 1)  # [4,1,1]
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
            tokens = {k: v.to(config.device) for k, v in tokens.items()}
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

        logger.info("\n=== Line-Level Metrics ===")
        logger.metric("recall@5", recall5, "line_level")
        logger.metric("precision@5", precision5, "line_level")
        logger.metric("effort@20%_LOC", effort20, "line_level")

    accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    cm = confusion_matrix(true_labels, predicted_labels)

    logger.info("\n--- Evaluation Metrics ---")
    logger.metric("accuracy", accuracy, "eval")
    logger.metric("precision", precision, "eval")
    logger.metric("recall", recall, "eval")
    logger.metric("f1", f1, "eval")
    logger.metric("accuracy", accuracy, "eval")

    logger.info("Confusion Matrix:")
    logger.info(cm)

    if(config.wandb):
        wandb.log({
        "eval/confusion_matrix": wandb.plot.confusion_matrix(
            preds=predicted_labels,
            y_true=true_labels,
            class_names=["invulnerable", "vulnerable"]
        )
    })