import random
from pathlib import Path
from random import shuffle

import PIL
import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from tts.base import BaseTrainer
from tts.logger.utils import plot_spectrogram_to_buf
from tts.utils import inf_loop, MetricTracker, ROOT_PATH


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics_train,
            metrics_test,
            optimizer,
            config,
            device,
            dataloader,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics_train, metrics_test, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config

        self.train_dataloader = dataloader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloader = dataloader
        self.lr_scheduler = lr_scheduler
        self.log_step = 100

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self._metrics_train], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self._metrics_test], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        names = ["length_target", "mel_target", "mel_pos", "src_seq",
                 "src_pos"]
        for tensor_for_gpu in names:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        for list_batch_idx, list_batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            for batch_idx, batch in enumerate(list_batch):
                try:
                    batch = self.process_batch(
                        batch,
                        batch_idx=batch_idx,
                        is_train=True,
                        metrics=self.train_metrics,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                if not (batch_idx + 1) % self.config['trainer']['batch_acum']:
                    self.train_metrics.update("grad norm", self.get_grad_norm())
                if batch_idx % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(batch_idx), batch["loss"].item()
                        )
                    )
                    self.writer.add_scalar(
                        "learning rate", self.optimizer.param_groups[0]['lr']
                    )
                    # self._log_predictions(is_validation=False, **batch)
                    # todo add log audio
                    """self._log_audio(batch['mixed'][0],
                                    batch['short'][0],
                                    batch['long'][0],
                                    batch['target'][0],
                                    batch['ref'][0],
                                    16000)"""
                    self._log_scalars(self.train_metrics)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()
                if batch_idx >= self.len_epoch:
                    break
        log = last_train_metrics

        val_log = self._evaluation_epoch(epoch, self.evaluation_dataloader)
        log.update(**{f"evaluation": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, batch_idx: int, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        print("BATCH:", batch.keys())
        print("OUTPUTS:", outputs.keys())
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        mel_loss, duration_predictor_loss, energy_predictor_loss, pitch_predictor_loss = self.criterion(
                                       batch['output'], batch['duration_pred'],
                                       batch['energy_pred'], batch['pitch_pred'],
                                       batch['mel_target'], batch['length_target'],
                                       batch['energy_target'], batch['pitch_target'])
        batch["loss"] = (mel_loss + duration_predictor_loss + energy_predictor_loss + pitch_predictor_loss) / 4
        batch["mel_loss"] = mel_loss
        batch["duration_predictor_loss"] = duration_predictor_loss
        batch["energy_predictor_loss"] = energy_predictor_loss
        batch["pitch_predictor_loss"] = pitch_predictor_loss
        if is_train and not ((batch_idx + 1) % self.config['trainer']['batch_acum']):
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            self.lr_scheduler.step()  # fix

        if not is_train or ((batch_idx + 1) % self.config['trainer']['batch_acum']):
            for loss_name in ("loss", "mel_loss", "duration_predictor_loss", "energy_predictor_loss",
                              "pitch_predictor_loss"):
                metrics.update(loss_name, batch[loss_name].item())
        return batch

    def _evaluation_epoch(self, epoch, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc="evaluation",
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    batch_idx=batch_idx,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )

            self.writer.set_step(epoch * self.len_epoch, "evaluation")
            # todo add log audio and spectrograms
            """self._log_audio(batch['mixed'][0],
                            batch['short'][0],
                            batch['long'][0],
                            batch['target'][0],
                            batch['ref'][0],
                            16000)"""
            self._log_scalars(self.evaluation_metrics)

            torch.save(self.model.state_dict(), ROOT_PATH / f"outputs/{epoch}.pth")

            # self._log_predictions(is_validation=True, **batch)
            # self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    def _log_audio(self, mixed, short, long, target, reference, sample_rate):
        if self.writer is None:
            return
        self.writer.add_audio("mixed", mixed, sample_rate)
        self.writer.add_audio("short", short, sample_rate)
        self.writer.add_audio("long", long, sample_rate)
        self.writer.add_audio("target", target, sample_rate)
        self.writer.add_audio("reference", reference, sample_rate)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
