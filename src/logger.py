import logging

class WandbHandler(logging.Handler):
    def __init__(self, wandb_run):
        super().__init__()
        self.wandb = wandb_run

    def emit(self, record):
        if hasattr(record, "wandb_data"):
            self.wandb.log(record.wandb_data, step=getattr(record, "step", None))

def setup_logger(name="clinescan", log_to_wandb=False, wandb_config=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if log_to_wandb:
            import wandb
            wandb_run = wandb.init(**(wandb_config or {}))
            wh = WandbHandler(wandb_run)
            wh.setLevel(logging.INFO)
            logger.addHandler(wh)
            logger.wandb_run = wandb_run

    return logger
