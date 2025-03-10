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
tokens = tokenizer(code_sample, padding=True, truncation=True, max_length=512, return_tensors="pt")
tokens = {key: value.to(device) for key, value in tokens.items()}

with torch.no_grad():
    outputs = model(**tokens)

print("Model shape", outputs.last_hidden_state.shape)

cls_embedding = outputs.last_hidden_state[:, 0, :] 
print("Function Embedding Shape:", cls_embedding.shape)