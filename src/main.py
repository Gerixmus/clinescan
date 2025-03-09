import json

with open("data.json", "r") as f:
    data = json.load(f)

import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)

print("Code:\n", data[0]["code"])
code_sample = data[0]["code"]
tokens = tokenizer(code_sample, padding=True, truncation=True, max_length=512)

print("Tokenized Input IDs:\n", tokens["input_ids"]) 