import optuna
import torch
import torch.nn as nn
import wandb
from transformers import RobertaModel

from training import train, evaluate

def objective_factory(config, train_samples, eval_samples, tokenizer, logger):
    """
    Creates an Optuna objective function with access to external variables
    by using a factory closure.
    """
    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])

        model = RobertaModel.from_pretrained(config.model_name).to(config.device)
        model.eval()

        classifier = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        ).to(config.device)

        params = list(model.parameters()) + list(classifier.parameters())

        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(params, lr=lr)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(params, lr=lr)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

        # loss_fn_name = trial.suggest_categorical("loss_fn", ["bce", "mse", "smooth_bce"])
        # if loss_fn_name == "bce":
        #     loss_fn = nn.BCEWithLogitsLoss()
        # elif loss_fn_name == "mse":
        #     loss_fn = nn.MSELoss()
        # elif loss_fn_name == "smooth_bce":
        #     loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.95], device=config.device))

        train(model, train_samples, config, tokenizer, classifier, optimizer, logger)
        f1 = evaluate(model, eval_samples, config, tokenizer, classifier, logger)

        return f1

    return objective


def tune(config, train_samples, eval_samples, tokenizer, logger):
    """
    Run Optuna tuning for classifier hyperparameters.
    """
    study = optuna.create_study(
        direction="maximize",
        study_name="vuln-detection-tuning",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    objective = objective_factory(config, train_samples, eval_samples, tokenizer, logger)
    study.optimize(objective, n_trials=10)

    print("\nBest trial:")
    print(study.best_trial.params)

    return study