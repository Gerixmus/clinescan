import torch
from torch import amp
import torch.nn.functional as F
from function import Function
from logger import WandbLogger
from config import Config

def train(
        model: torch.nn.Module,
        train_samples: list[Function],
        config: Config,
        tokenizer, 
        classifier: torch.nn.Module,
        optimizer, 
        logger: WandbLogger
    ) -> None:

    model.train()
    classifier.train()
    scaler = amp.GradScaler(config.device.type)

    for epoch in range(config.epochs):
        for i, item in enumerate(train_samples):
            code = "\n".join(item.code)
            label = torch.tensor([item.vul], dtype=torch.float32).to(config.device) 

            tokens = tokenizer(
                code, padding=True, truncation=True, max_length=512,
                return_tensors="pt", return_offsets_mapping=True
            )
            offset_mapping = tokens.pop("offset_mapping")[0].tolist()
            tokens = {k: v.to(config.device) for k, v in tokens.items()}

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

            logger.debug(f"Epoch {epoch} Sample {i} | Loss: {loss.item():.4f}")
            logger.metric("loss", loss.item(), "train", epoch * len(train_samples) + i)