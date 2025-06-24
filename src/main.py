import torch
import torch.nn as nn
import random
from transformers import RobertaModel, RobertaTokenizerFast
from data import get_train_test_split
import random
from logger import setup_logger
from training import train, evaluate
from config import Config
from tuning import tune

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

TRAIN = False

config = Config(
    epochs=1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed=42,
    sample_size=0.01,
    learning_rate=1e-5,
    batch_size=1,
    model_name="microsoft/codebert-base",
    wandb=False
    )

tokenizer = RobertaTokenizerFast.from_pretrained(config.model_name)
model = RobertaModel.from_pretrained(config.model_name, attn_implementation="eager").to(config.device)

set_seed(seed=config.seed)

logger = setup_logger("clinescan", config)

train_samples, eval_samples = get_train_test_split(config, logger)

classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 1)
).to(config.device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(classifier.parameters()), lr=config.learning_rate
)

if(TRAIN):
    train(model, train_samples, config, tokenizer, classifier, optimizer, logger)
    evaluate(model, eval_samples, config, tokenizer, classifier, logger)
else:
    tune(config, train_samples, eval_samples, tokenizer, logger)
