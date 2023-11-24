import argparse
import collections
import warnings

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
import torch

from tts.trainer import Trainer
from tts.utils import prepare_device, get_logger
from tts.utils.object_loading import get_dataloaders

# Отключение предупреждений
warnings.simplefilter("ignore", UserWarning)

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


@hydra.main(config_path='tts/configs', config_name='main_config')
def main(clf: DictConfig):
    logger = get_logger("train")
    # setup data_loader instances
    dataloader = get_dataloaders(clf['data'])
    # build model architecture, then print to console
    model = instantiate(clf["arch"])
    logger.info(model)
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(clf["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    # get function handles of loss and metrics
    loss_module = instantiate(clf["loss"]).to(device)
    metrics_test = []
    metrics_train = [
    ]
    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(clf["optimizer"], trainable_params)
    lr_scheduler = instantiate(clf["lr_scheduler"], optimizer)
    trainer = Trainer(
        model,
        loss_module,
        metrics_train,
        metrics_test,
        optimizer,
        config=clf,
        device=device,
        dataloader=dataloader,
        lr_scheduler=lr_scheduler,
        len_epoch=clf["trainer"].get("len_epoch", None)
    )
    trainer.train()


if __name__ == "__main__":
    main()
