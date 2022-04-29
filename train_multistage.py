# @Author: yican, yelanlan
# @Date: 2020-06-16 20:36:19
# @Last Modified by:   yican.yc
# @Last Modified time: 2020-06-16 20:36:19
# Standard libraries
import os
import gc
from pathlib import Path
from pydoc import classname
from time import time, sleep
import traceback
from typing import Dict
import numpy as np
import time as ts
from pandas import DataFrame
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchcam.methods.activation import CAM
from torchvision.transforms.functional import to_tensor
# Third party libraries
import torch
from torchcam.methods.gradient import SmoothGradCAMpp
from datasets.dataset import ProjectDataModule
from sklearn.model_selection import KFold

# User defined libraries
from models import se_resnext50_32x4d, se_resnext50_32x4d_arcface, se_resnext50_32x4d_mask, se_resnext50_32x4d_mask2
from utils import init_hparams, init_logger, load_training_data, seed_reproducer, load_data
from loss_function import AngularPenaltySMLoss, CrossEntropyLossOneHot, FocalLoss
from lrs_scheduler import WarmRestart, warm_restart
from utils.common import select_fn_indexes, visualization
from PIL import Image
from utils import *
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import AveragePrecision, AUROC
from torch import Tensor, embedding
from filelock import FileLock
from train import CoolSystem


class BinaryAveragePrecision(AveragePrecision):

    def __init__(self,
                 num_classes: Optional[int] = None,
                 pos_label: Optional[int] = None,
                 average: Optional[str] = "macro",
                 compute_on_step: Optional[bool] = None,
                 **kwargs: Dict[str, Any]) -> None:
        super().__init__(num_classes, pos_label, average, compute_on_step,
                         **kwargs)

    def binarization(self, label: Tensor) -> None:
        return torch.cat(
            [label[:, [0]], label[:, 1:].sum(axis=1, keepdim=True)], axis=1)

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = self.binarization(preds)
        target = self.binarization(target)
        return super().update(preds, target)


class MultiStageCoolSystem(CoolSystem):

    def __init__(self, hparams):
        super().__init__(hparams)


if __name__ == "__main__":
    # Make experiment reproducible
    seed_reproducer(2022)

    # Init Hyperparameters
    hparams = init_training_config()
    # init logger
    logger = init_logger("HEC", log_dir=hparams.log_dir)
    os.environ["EXP_DIR"] = hparams.log_dir
    hparams.HEC_LOGGER = logger
    # Do cross validation
    try:
        for fold_i in range(5):
            hparams.fold_i = fold_i
            if is_skip_current_fold(fold_i, hparams):
                logger.info(f'Skipped fold {fold_i}')
                continue
            tf_logger = TensorBoardLogger(
                os.path.join(hparams.log_dir, f'fold-{fold_i}'))
            da = ProjectDataModule(hparams)
            # Define callbacks
            checkpoint_path = os.path.join(hparams.log_dir, 'checkpoints')
            # os.makedirs(checkpoint_path, exist_ok=True)
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_path,
                monitor="val_roc_auc",
                save_top_k=hparams.save_top_k,
                mode="max",
                filename=f"fold={fold_i}" +
                "-{epoch}-{val_loss:.4f}-{val_roc_auc:.4f}")
            checkpoint_callback.CHECKPOINT_NAME_LAST = f"latest-fold={fold_i}" + "-{epoch}-{val_loss:.4f}-{val_roc_auc:.4f}"
            # other_checkpoint_callback = ModelCheckpoint(
            #     dirpath=checkpoint_path,
            #     monitor="other_roc_auc",
            #     save_top_k=2,
            #     mode="max",
            #     filename=f"fold={fold_i}" +
            #     "-[test-real-world]-{epoch}-{other_loss:.3f}-{other_roc_auc:.4f}"
            # )
            early_stop_callback = EarlyStopping(monitor="val_roc_auc",
                                                patience=hparams.patience,
                                                mode="max",
                                                verbose=True)

            # Instance Model, Trainer and train model
            model = CoolSystem(hparams)
            if hparams.debug:
                logger.info(model)
            trainer = pl.Trainer(
                logger=tf_logger,
                replace_sampler_ddp=False,
                fast_dev_run=hparams.debug,
                strategy=get_training_strategy(hparams),
                gpus=hparams.gpus,
                num_nodes=1,
                min_epochs=hparams.min_epochs,
                max_epochs=hparams.max_epochs,
                # val_check_interval=1,
                callbacks=[
                    early_stop_callback,
                    checkpoint_callback  # other_checkpoint_callback
                ],
                precision=hparams.precision,
                num_sanity_val_steps=0,
                profiler=False,
                gradient_clip_val=hparams.gradient_clip_val)
            if hparams.eval_mode == 'train':
                trainer.fit(model,
                            datamodule=da,
                            ckpt_path=get_checkpoint_resume(hparams))
                try:
                    trainer.test(ckpt_path=checkpoint_callback.best_model_path,
                                 datamodule=da)
                    # valid_roc_auc_scores.append(
                    # round(checkpoint_callback.best_model_score, 4))
                except Exception as e:
                    traceback.print_exc()
                    print('Proccessing wrong in testing.')
            elif hparams.eval_mode == 'test':
                trainer.test(model=model,
                             ckpt_path=get_checkpoint_resume(hparams),
                             datamodule=da)
        if model.global_rank == 0:
            post_embedding(hparams)
    except Exception as e:
        traceback.print_exc()
        raise e
    finally:
        if hparams.debug and hparams.clean_debug:
            shutil.rmtree(hparams.log_dir)
            logger.info(
                'Debugging mode, clean the output dir in the debug stage.')
