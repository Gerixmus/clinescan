import json
from function import Function
from sklearn.model_selection import train_test_split
import random
from config import Config

def get_entries() -> list[Function]:
    with open("data.json", "r") as f:
        raw_data = json.load(f)

    data = []
    for item in raw_data:
        code_lines = item["code"].split('\n')
        func = Function(
            code=code_lines,
            vul=item["vul"],
            flaw_line_no=item["flaw_line_no"],
            bigvul_id=item["bigvul_id"]
        )
        data.append(func)
    return data

def get_train_test_split(config: Config, logger):
    all_data = get_entries()
    labels = [entry.vul for entry in all_data]

    all_train_samples, eval_samples = train_test_split(
        all_data,
        test_size = config.sample_size * 0.2,
        random_state = config.seed,
        shuffle = True,
        stratify = labels
    )

    vul_train = [x for x in all_train_samples if x.vul == 1]
    nonvul_train = [x for x in all_train_samples if x.vul == 0]

    balanced_train_size = int(len(all_data) * config.sample_size * 0.8)

    half_size = balanced_train_size // 2

    vul_train_balanced = random.sample(vul_train, min(len(vul_train), half_size))
    nonvul_train_balanced = random.sample(nonvul_train, min(len(nonvul_train), half_size))

    train_samples = vul_train_balanced + nonvul_train_balanced
    random.shuffle(train_samples)

    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Eval samples: {len(eval_samples)}")

    return train_samples, eval_samples
