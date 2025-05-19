import json

def get_entries(amount: int):
    with open("data.json", "r") as f:
        data = json.load(f)

    pos_samples = [item for item in data if item["vul"] == 1]
    neg_samples = [item for item in data if item["vul"] == 0]

    print(f"Vulnerable samples: {len(pos_samples)}, Normal samples: {len(neg_samples)}")

    train_samples = pos_samples[:amount] + neg_samples[:amount]
    used_indices = set()

    for item in train_samples:
        used_indices.add(data.index(item))

    remaining_indices = [i for i in range(len(data)) if i not in used_indices]

    return data, train_samples, used_indices, remaining_indices