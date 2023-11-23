# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from fairseq2.models.transformer import (
    TransformerDecoderModel)

from .llama_dataloader import LLaMADataLoader, LLaMABatch

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.optim import Adam

from fairseq2.models.sequence import SequenceModelOutput,SequenceBatch
from fairseq2.optim.lr_scheduler import MyleLR
from fairseq2.typing import Device


logger = logging.getLogger(__name__)


@dataclass
class FinetuneParams:
    save_model_path: Path
    """Path were to save finetuned model."""

    max_epochs: int = 10
    """ Maximum number of training epochs"""

    warmup_steps: int = 100
    """ Number of steps with linearly increasing LR"""

    label_smoothing: float = 0.2
    """ Label smoothing coefficient for nll_loss """

    log_steps: int = 10
    """ Log inner loss after each `log_steps` training steps"""

    eval_steps: int = 50
    """ Get eval loss after each `eval_steps` training steps """

    patience: int = 3
    """ Terminate if eval loss did not improve
    over the last `patience * eval_steps` training steps"""

    learning_rate: float = 1e-5
    """ Optimizer learining rate """

    train_batch_size: int = 5
    """The batch size during train steps"""

    eval_batch_size: int = 5
    """The batch size during evaluation."""

    device: Device = torch.device("cuda")
    """ Where to run computation"""


class LossCollector:
    """Aggregrates loss history across nodes"""

    def __init__(self, device: Optional[Device] = None, reduce_op: str = "avg"):
        self.n_samples: float = 0
        self.val_sum: float = 0.0
        self.reduce_op = reduce_op
        self.device = device

    def reset(self) -> None:
        self.n_samples = 0
        self.val_sum = 0.0

    def update(self, n_samples: int, batch_loss: float) -> None:
        self.n_samples += n_samples
        self.val_sum += batch_loss

    def reduce(self) -> float:
        if self.reduce_op == "avg":
            return self.val_sum / (self.n_samples + 1)
        if self.reduce_op == "sum":
            return self.val_sum
        raise ValueError()



class LLaMAFinetune:
    def __init__(
        self,
        model: TransformerDecoderModel,
        params: FinetuneParams,
        train_data_loader: LLaMADataLoader,
        eval_data_loader: Optional[LLaMADataLoader] = None,
    ):
        self.params = params
        self.model = model
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.params.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-08,
            maximize=False,
            weight_decay=0.0,
            fused=True,
        )
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.lr_scheduler = MyleLR(
            optimizer=self.optimizer,
            num_warmup_steps=self.params.warmup_steps,
            start_lr=1e-9,
        )

        self.train_loss_hist = LossCollector(device=params.device)
        self.epoch_idx: int = 0
        self.update_idx: int = 0
        self.patience_left: int = self.params.patience
        self.best_eval_loss: Optional[float] = None
        self.is_best_state: bool = False

    def _reset_stats(self) -> None:
        self.train_loss_hist.reset()
        self.epoch_idx = 0
        self.update_idx = 0
        self.patience_left = self.params.patience
        self.best_eval_loss = None
        self.is_best_state = False

    def _update_eval_stats(self, eval_loss: float) -> None:
        self.is_best_state = (
            self.best_eval_loss is None or eval_loss < self.best_eval_loss
        )
        self.best_eval_loss = eval_loss if self.is_best_state else self.best_eval_loss
        self.patience_left = (
            self.params.patience if self.is_best_state else self.patience_left - 1
        )
        logger.info(
            f"Eval after {self.update_idx} updates: "
            f"loss={eval_loss:.4f} "
            f"best_loss={self.best_eval_loss:.4f} "
            f"patience_steps_left={self.patience_left}"
        )

    def _eval_model(self) -> None:
        """Calc avg loss on eval dataset and update evaluation stats"""
        if self.eval_data_loader is None:
            return
        logger.info("Run evaluation")
        self.model.eval()
        eval_loss = None
        with torch.no_grad():
            for batch in self.eval_data_loader.get_dataloader():
                output = self.model(batch.data_tokens)
                loss = output.compute_loss(
                    batch.target_tokens,
                    label_smoothing=self.params.label_smoothing
                    )
                if loss.isnan():
                    logger.warning("Eval loss value is NaN, setting to inf")
                    eval_loss = float("Inf")
                else:
                    eval_loss = loss.item()
                del batch  # force memory release
        self._update_eval_stats(eval_loss)

    def _train_step_log(self):
        """Log train stats"""
        if (self.update_idx + 1) % self.params.log_steps == 0:
            avg_loss = self.train_loss_hist.reduce()
            self.train_loss_hist.reset()
            logger.info(
                f"Epoch {str(self.epoch_idx + 1).zfill(3)} / "
                f"update {str(self.update_idx + 1).zfill(5)}: "
                f"train loss={avg_loss:.4f} "
                f"last lr={self.lr_scheduler.get_last_lr()[0]:.2E}"
            )

    def _train_step(self, batch: LLaMABatch) -> None:
        """Run one train step"""
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(batch.data_tokens)
        loss = output.compute_loss(
            batch.target_tokens,
            label_smoothing=self.params.label_smoothing
            )
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.lr_scheduler.step()
        self.train_loss_hist.update(1, loss.item())
        self._train_step_log()

    def _save_model(self):
        logger.info("Saving model")
        state_dict = {
            key.replace("module.model.", ""): value
            for key, value in self.model.state_dict().items()
        }
        torch.save(state_dict, self.params.save_model_path)

    def run(self):
        logger.info("Start finetuning")
        self._reset_stats()
        self._eval_model()
        batch_itr = self.train_data_loader.get_dataloader()
        while self.epoch_idx < self.params.max_epochs and self.patience_left:
            for train_batch in batch_itr:
                self._train_step(batch=train_batch)
                if self.update_idx and self.update_idx % self.params.eval_steps == 0:
                    self._eval_model()
                    if self.is_best_state:
                        self._save_model()
                    elif not self.patience_left:
                        no_improve_steps = self.params.eval_steps * self.params.patience
                        logger.info(
                            "Early termination, as eval loss did not improve "
                            f"over last {no_improve_steps} updates"
                        )
                        break
                logger.info('Epoch: {} Batch: {}'.format(self.epoch_idx,self.update_idx))
                self.update_idx += 1
            self.epoch_idx += 1