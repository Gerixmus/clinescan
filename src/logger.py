import logging
import types

class WandbLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.wandb_run = None 

    def set_wandb_run(self, wandb_run):
        self.wandb_run = wandb_run

    def metric(self, name: str, value, level: str, step: int = None):
        self.info(f"{name}: {value:.4f}")
        if self.wandb_run:
            log_dict = {f"{level}/{name}": value}
            if step is not None:
                log_dict["step"] = step
            self.wandb_run.log(log_dict)

def setup_logger(name="clinescan", log_to_wandb=False, wandb_config=None):
    logging.setLoggerClass(WandbLogger)
    logger: WandbLogger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if log_to_wandb:
        import wandb
        wandb_run = wandb.init(**(wandb_config or {}))
        logger.set_wandb_run(wandb_run)

    return logger