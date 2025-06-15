import logging
from config import Config

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

def setup_logger(name, config: Config):
    logging.setLoggerClass(WandbLogger)
    logger: WandbLogger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if config.wandb:
        import wandb
        wandb_run = wandb.init(
            project = "clinescan-eval",
            name = f"{config.epochs}x-{config.sample_size*100}%",
            config = {
                "model": config.model_name,
                "lr": config.learning_rate,
                "batch_size": config.batch_size,
                "sample_size": config.sample_size
            }
        )
        logger.set_wandb_run(wandb_run)

    return logger