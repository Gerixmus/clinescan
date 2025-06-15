import torch
import torch.nn as nn
import random
from transformers import RobertaModel, RobertaTokenizerFast
from data import get_train_test_split
from torch import amp
import random
from evaluate import evaluate
from logger import setup_logger
from training import train

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED = 42
EPOCHS = 3
SAMPLE_SIZE = 0.05
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/codebert-base"
WANDB = False

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
model = RobertaModel.from_pretrained(MODEL_NAME, attn_implementation="eager").to(DEVICE)

set_seed(seed=SEED)

logger = setup_logger(
    name="clinescan",
    log_to_wandb=WANDB,
    wandb_config={
        "project": "clinescan-eval",
        "name": f"{EPOCHS}x-{SAMPLE_SIZE*100}%",
        "config": {
            "model": MODEL_NAME,
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "sample_size": SAMPLE_SIZE
        }
    }
)

train_samples, eval_samples = get_train_test_split(SAMPLE_SIZE, SEED, logger)



classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 1)
).to(DEVICE)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(classifier.parameters()), lr=LEARNING_RATE
)

train(model, train_samples, EPOCHS, DEVICE, tokenizer, optimizer, classifier, logger)
evaluate(model, eval_samples, DEVICE, WANDB, tokenizer, logger, classifier)