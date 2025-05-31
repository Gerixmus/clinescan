import torch
import torch.nn as nn
import random

from transformers import RobertaTokenizer, RobertaModel

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import optuna
import wandb
# wand dashboard
# https://wandb.ai/comcatmangreen-university-klagenfurt-/vuldetection-optimization/groups/vuldetection-optim1/workspace

import data

# very clean code
# https://github.com/nzw0301/optuna-wandb/blob/main/part-1/wandb_optuna.py
# Possible hyperparameters to tune
#   - layers
#   - neurons
#   - learning rate
#   - optimizer
#   - loss function


# Global parameters external to model
TOKENIZER = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
MODEL = RobertaModel.from_pretrained("microsoft/codebert-base")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STUDY_NAME = "vuldetection-optim1"
EPOCHS = 3

def objective(trial):
    # Hyperparameter selection
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-2)
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW", "RMSprop"])
    loss_function = trial.suggest_categorical("loss_function", ["L1Loss", "NLLLoss", "CrossEntropyLoss"])

    # Model initialization
    classifier = nn.Sequential(
        nn.Linear(768, 2)
    ).to(DEVICE)


    # Optimizer initialization
    # Python doesn't have switch case lol
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.RMSprop(classifier.parameters(), lr=lr, weight_decay=weight_decay)

    if loss_function == "L1Loss":
        loss_fn = nn.L1Loss()
    elif loss_function == "MSELoss":
        loss_fn = nn.MSELoss()
    elif loss_function == "NLLLoss":
        loss_fn = nn.NLLLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)

    # init tracking experiment.
    # hyper-parameters, trial id are stored.
    config = dict(trial.params)
    config["trial.number"] = trial.number
    wandb.init(
        project="project",
        config=config,
        entity="user-organization",
        group=STUDY_NAME,
        reinit=True,
    )
    # Training of the model.
    for epoch in range(EPOCHS):

        # Training loop
        classifier = modelTrain(classifier, loss_fn, optimizer)
        accuracy = modelEval(classifier)

        trial.report(accuracy, epoch)

        # report validation accuracy to wandb
        wandb.log(data={"validation accuracy": accuracy}, step=epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            wandb.run.summary["state"] = "pruned"
            wandb.finish(quiet=True)
            raise optuna.exceptions.TrialPruned()



    # Handle pruning based on the intermediate value.
    #if trial.should_prune():
    #        wandb.run.summary["state"] = "pruned"
    #        wandb.finish(quiet=True)
    #        raise optuna.exceptions.TrialPruned()

    # report the final validation accuracy to wandb
    wandb.run.summary["final accuracy"] = accuracy
    wandb.run.summary["state"] = "completed"
    wandb.finish(quiet=True)


    return accuracy



def modelTrain(classifier, loss_fn, optimizer):
    for i, item in enumerate(train_samples):
        code = item["code"]
        label = torch.tensor([item["vul"]]).to(DEVICE)

        tokens = TOKENIZER(code, padding=True, truncation=True, max_length=512, return_tensors="pt")
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = MODEL(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        logits = classifier(cls_embedding)
        loss = loss_fn(logits, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print(f"Sample {i} | Label: {item['vul']} | Loss: {loss.item():.4f}")

    return classifier

def modelEval(classifier):
    correct = 0


    for i in eval_indices:
        code = all_data[i]["code"]
        true_label = all_data[i]["vul"]

        tokens = TOKENIZER(code, padding=True, truncation=True, max_length=512, return_tensors="pt")
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = MODEL(**tokens)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            logits = classifier(cls_embedding)

        predicted = logits.argmax(dim=1).item()

        match = "✅" if predicted == true_label else "❌"
        if predicted == true_label:
            correct += 1

        # print(f"[{match}] Index: {i} | Predicted: {predicted} | True: {true_label}")

    print(f"\nAccuracy: {correct}/{len(train_samples)} = {100 * correct / len(train_samples):.2f}%")

    accuracy = 100 * correct / len(train_samples)
    # report validation accuracy to wandb
    # wandb.log(data={"validation accuracy": accuracy})

    return accuracy

# Data
wandb.login(key='key')


all_data, train_samples, used_indices, remaining_indices = data.get_entries(100)
eval_indices = random.sample(remaining_indices, min(len(train_samples), len(remaining_indices)))

MODEL.to(DEVICE)
MODEL.eval()


study = optuna.create_study(
    direction="maximize",
    study_name=STUDY_NAME,
    pruner=optuna.pruners.MedianPruner(),
)

study.optimize(objective, n_trials=10)
print(optuna.importance.get_param_importances(study))

print("Best trial:")
print(study.best_trial.params)
