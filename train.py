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


class CoolSystem(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        # for key in hparams.keys():
        #     self.hparams[key] = hparams[key]
        self.hparams.update(hparams)
        # 让每次模型初始化一致, 不让只要中间有再次初始化的情况, 结果立马跑偏
        seed_reproducer(self.hparams.seed)
        self.num_classes = hparams.num_classes
        if self.hparams.model_type == 'base':
            self.model = se_resnext50_32x4d(num_classes=self.num_classes)
        elif self.hparams.model_type == 'mask':
            self.model = se_resnext50_32x4d_mask(num_classes=self.num_classes)
        elif self.hparams.model_type == 'mask2':
            self.model = se_resnext50_32x4d_mask2(num_classes=self.num_classes)
        elif self.hparams.model_type == 'arcface':
            self.model = se_resnext50_32x4d_arcface(num_classes=self.num_classes)
        
        if self.hparams.loss == 'cross_entropy_loss':
            self.criterion = CrossEntropyLossOneHot()
        elif self.hparams.loss == 'focal_loss':
            self.criterion = FocalLoss(alpha=[1] * self.num_classes,
            num_classes=self.num_classes)
        elif self.hparams.loss == 'arcface_loss':
            self.criterion = AngularPenaltySMLoss()
        self.HEC_LOGGER = init_logger('HEC', hparams.log_dir)
        # create directories for testing
        self.test_output_dir = os.path.join(hparams.log_dir,
                                            f'fold-{hparams.fold_i}', 'test')
        os.makedirs(self.test_output_dir, exist_ok=True)
        self.test_vis_output = os.path.join(self.test_output_dir, 'vis')
        os.makedirs(self.test_vis_output, exist_ok=True)
        self.test_cat_output = os.path.join(self.test_output_dir, 'cat')
        os.makedirs(self.test_cat_output, exist_ok=True)
        self.test_set_output = os.path.join(self.test_output_dir, 'set')
        os.makedirs(self.test_set_output, exist_ok=True)

        # self.test_selection_record_path = os.path.join(
        #     os.path.join(self.test_output_dir, 'selection.csv'))
        # self.test_performance_record_path = os.path.join(
        #     self.test_output_dir, 'performance.csv')
        self.test_set_num = self.hparams.test_real_world_num + 1

        for test_dataloader_idx in range(self.test_set_num):
            test_data_output_dir = os.path.join(self.test_set_output,
                                                str(test_dataloader_idx))
            os.makedirs(test_data_output_dir, exist_ok=True)
            self.test_pred_path = os.path.join(test_data_output_dir,
                                               'pred.csv')
            self.test_label_path = os.path.join(test_data_output_dir,
                                                'label.csv')
            self.create_empty_csv(self.test_pred_path,
                                  self.hparams.header_names)
            self.create_empty_csv(self.test_label_path,
                                  self.hparams.header_names)

        # self.test_selection_records = self.load_records(
        #     self.test_selection_record_path, MODEL_SELECTION_RECORD_HEADERS)
        # self.test_performance_records = self.load_records(
        #     self.test_performance_record_path, PERFORMANCE_RECORD_HEADERS)

        # create directories for validation
        self.val_output_dir = os.path.join(hparams.log_dir,
                                           f'fold-{hparams.fold_i}', 'val')
        os.makedirs(self.val_output_dir, exist_ok=True)
        self.val_vis_output = os.path.join(self.val_output_dir, 'vis')
        os.makedirs(self.val_vis_output, exist_ok=True)
        self.val_cat_output = os.path.join(self.val_output_dir, 'cat')
        os.makedirs(self.val_cat_output, exist_ok=True)
        self.val_selection_record_path = os.path.join(
            os.path.join(self.val_output_dir, 'selection.csv'))
        self.val_performance_record_path = os.path.join(
            self.val_output_dir, 'performance.csv')
        self.val_selection_records = self.load_records(
            self.val_selection_record_path, MODEL_SELECTION_RECORD_HEADERS)
        self.val_performance_records = self.load_records(
            self.val_performance_record_path, PERFORMANCE_RECORD_HEADERS)
        # self.cam_extractors = [SmoothGradCAMpp(self, target_layer=f'model.model_ft.{i}.0.downsample') for i in range(1, 5)]
        # self.cam_extractors = [
        #                         SmoothGradCAMpp(self, target_layer=f'model.model_ft.0'),
        #                         SmoothGradCAMpp(self, target_layer=f'model.model_ft.1.2'),
        #                         SmoothGradCAMpp(self, target_layer=f'model.model_ft.2.3'),
        #                         SmoothGradCAMpp(self, target_layer=f'model.model_ft.3.5'),
        #                         SmoothGradCAMpp(self, target_layer=f'model.model_ft.4.2'),
        #                        ]
        # self.cam_extractors = [SmoothGradCAMpp(self, target_layer=f'model.model_ft.{i}.2.se_module') for i in range(1, 5)]
        # self.cam_extractors = [CAM(self)]
        self.cam_extractors = [
            CAM(self,
                fc_layer='model.binary_head.fc.0',
                target_layer='model.model_ft.4.0.se_module'),
            CAM(self,
                fc_layer='model.binary_head.fc.0',
                target_layer='model.model_ft.4.1.se_module'),
            CAM(self,
                fc_layer='model.binary_head.fc.0',
                target_layer='model.model_ft.4.2.se_module'),
        ]

        if self.hparams.metrics == 'roc_auc':
            self.metric = AUROC(num_classes=self.hparams.num_classes)
        elif self.hparams.metrics == 'mAP':
            self.metric = AveragePrecision(
                num_classes=self.hparams.num_classes)
        elif self.hparams.metrics == 'bn_mAP':
            self.metric = BinaryAveragePrecision(pos_label=0)
        self.beta = torch.distributions.Beta(torch.tensor([1.2]), torch.tensor([0.2]))
        self.lr = self.hparams.lr
    
    def get_learning_rate(self):
        if hasattr(self, 'scheduler'):
            return self.scheduler.get_lr()[0]
        else:
            return self.lr if hasattr(self, 'scheduler') else self.hparams.lr

    def load_records(self, record_path, columns):
        if os.path.exists(record_path):
            return pd.read_csv(record_path)
        else:
            df = pd.DataFrame(columns=columns)
            df.to_csv(record_path, index=False)
            return df

    def create_empty_csv(self, record_path, columns):
        if os.path.exists(record_path):
            os.remove(record_path)
        df = pd.DataFrame(columns=columns)
        df.to_csv(record_path, index=False)
        return df

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.hparams.lr,
                                          betas=(0.9, 0.999),
                                          eps=1e-08,
                                          weight_decay=0)
        self.scheduler = WarmRestart(self.optimizer,
                                     T_max=10,
                                     T_mult=1,
                                     eta_min=1e-5)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        step_start_time = time()
        images, roi_mask, ar_mask, labels, data_load_time, _ = batch

        scores = self(images, roi_mask=roi_mask, ar_mask=ar_mask)
        loss = self.criterion(scores, labels)
        if torch.rand(1) < 0.5:
            images, labels = mixup(images, labels, self.beta.sample().to(images.device))
        data_load_time = torch.sum(data_load_time)

        return {
            "loss":
            loss,
            "data_load_time":
            data_load_time,
            "batch_run_time":
            torch.Tensor([time() - step_start_time + data_load_time
                          ]).to(data_load_time.device),
        }

    def training_epoch_end(self, outputs):
        # outputs is the return of training_step
        train_loss_mean = torch.stack([output["loss"]
                                       for output in outputs]).mean()
        self.data_load_times = torch.stack(
            [output["data_load_time"] for output in outputs]).sum()
        self.batch_run_times = torch.stack(
            [output["batch_run_time"] for output in outputs]).sum()

        # self.HEC_LOGGER_kun.info(self.current_epoch)
        if self.current_epoch + 1 < (self.trainer.max_epochs - 4) and hasattr(self, 'scheduler'):
            self.scheduler = warm_restart(self.scheduler, T_mult=2)

        # return {"train_loss": train_loss_mean}

    # def test_step(self, batch, batch_idx):
    #     step_start_time = time()
    #     with torch.enable_grad():
    #         # self.unfreeze()
    #         self.eval()
    #         self.zero_grad()
    #         images, labels, data_load_time, filenames = batch
    #         images.requires_grad = True
    #         # self.HEC_LOGGER_kun.info(f'{dataloader_idx}: {images.size()}')
    #         data_load_time = torch.sum(data_load_time)
    #         scores = self(images)
    #         visualization(batch_idx,
    #                       self.cam_extractors,
    #                       images,
    #                       scores,
    #                       labels,
    #                       filenames,
    #                       os.path.join(self.vis_test_output,
    #                                    str(self.current_epoch)),
    #                       save_batch=False,
    #                       save_per_image=False,
    #                       mean=self.hparams.norm['mean'],
    #                       std=self.hparams.norm['std'])
    #         loss = self.criterion(scores, labels)
    #         # self.freeze()
    #     # must return key -> val_loss
    #     return {
    #         "filenames":
    #         np.array(filenames),
    #         "test_loss":
    #         loss.detach(),
    #         "scores":
    #         scores.detach(),
    #         "labels":
    #         labels.detach(),
    #         "data_load_time":
    #         data_load_time,
    #         "batch_run_time":
    #         torch.Tensor([time() - step_start_time + data_load_time
    #                       ]).to(data_load_time.device),
    #     }

    def test_step(self, batch, batch_idx, dataloader_idx):
        step_start_time = time()
        images, roi_mask, ar_mask, labels, data_load_time, filenames = batch
        data_load_time = torch.sum(data_load_time)
        scores = self(images, roi_mask=roi_mask, ar_mask=ar_mask)
        loss = self.criterion(scores, labels)

        np_scores = torch.softmax(scores, dim=1).detach().cpu().numpy()
        np_labels = labels.detach().cpu().numpy()
        np_scores = pd.DataFrame(
            np.concatenate([np.array(filenames).reshape(-1, 1), np_scores],
                           axis=1))
        np_labels = pd.DataFrame(
            np.concatenate([np.array(filenames).reshape(-1, 1), np_labels],
                           axis=1))
        # set lock for avoiding concurrence problems
        test_data_output_dir = os.path.join(self.test_set_output,
                                            str(dataloader_idx))
        test_pred_path = os.path.join(test_data_output_dir, 'pred.csv')
        test_label_path = os.path.join(test_data_output_dir, 'label.csv')
        with FileLock(test_pred_path + '.lock'):
            np_scores.to_csv(test_pred_path,
                             mode='a',
                             header=False,
                             index=False)

        with FileLock(test_label_path + '.lock'):
            np_labels.to_csv(test_label_path,
                             mode='a',
                             header=False,
                             index=False)

        return {
            "filenames":
            np.array(filenames),
            "loss":
            loss.detach(),
            "scores":
            scores.detach(),
            "labels":
            labels.detach(),
            "data_load_time":
            data_load_time,
            "batch_run_time":
            torch.Tensor([time() - step_start_time + data_load_time
                          ]).to(data_load_time.device),
        }

    def test_epoch_end(self, outputs):
        # compute loss
        test_info = collect_distributed_info(self.hparams, outputs[0])
        other_info = collect_other_distributed_info(self.hparams,
                                                    outputs,
                                                    created=False)
        if self.global_rank == 0:
            sleep(3)  # wait other processes to complete inference
            for dataloader_idx in range(len(outputs)):
                test_data_output_dir = os.path.join(self.test_set_output,
                                                    str(dataloader_idx))
                test_pred_path = os.path.join(test_data_output_dir, 'pred.csv')
                test_label_path = os.path.join(test_data_output_dir,
                                               'label.csv')
                preds = pd.read_csv(test_pred_path)
                labels = pd.read_csv(test_label_path)
                preds = preds.sort_values(by=['filename'])
                labels = labels.sort_values(by=['filename'])
                assert len(preds) == len(labels)
                post_report(self.hparams,
                            self.current_epoch,
                            torch.from_numpy(preds.iloc[:, 1:].to_numpy()),
                            torch.from_numpy(labels.iloc[:, 1:].to_numpy()),
                            preds.iloc[:, [0]].to_numpy().reshape(-1),
                            test_data_output_dir,
                            ger_report=True)
        self.HEC_LOGGER.info(
            f"{self.hparams.fold_i}-{self.current_epoch} | "
            f"test_loss : {test_info.loss_mean:.4f} | "
            f"test_{self.hparams.metrics} : {test_info.metrics:.4f} | "
            f"data_load_times : {test_info.data_load_times:.2f} | "
            f"batch_run_times : {test_info.batch_run_times:.2f}")
        # self.log('test_loss', test_info.test_loss_mean)
        # self.log('test_metrics', test_info.test_roc_auc)

        # write_distributed_records(self.global_rank, self.hparams.fold_i,
        #                           self.current_epoch, self.global_step,
        #                           test_info, other_info,
        #                           self.test_selection_record_path,
        #                           self.test_performance_record_path)
        return {
            "test_loss": test_info.loss_mean,
            f"test_{self.hparams.metrics}": test_info.metrics
        }

    def validation_step(self, batch, batch_idx, dataloader_idx):
        step_start_time = time()
        if dataloader_idx == 0:
            # self.model.eval()
            # self.model.zero_grad()
            # self.eval()
            with torch.enable_grad():
                # self.unfreeze()
                self.eval()
                self.zero_grad()
                images, roi_mask, ar_mask, labels, data_load_time, filenames = batch
                images.requires_grad = True
                # self.HEC_LOGGER_kun.info(f'{dataloader_idx}: {images.size()}')
                data_load_time = torch.sum(data_load_time)
                scores = self(images, roi_mask=roi_mask, ar_mask=ar_mask)
                visualization(batch_idx,
                              self.cam_extractors,
                              images,
                              scores,
                              labels,
                              filenames,
                              os.path.join(self.val_vis_output,
                                           str(self.current_epoch)),
                              save_batch=False,
                              save_per_image=True,
                              mean=self.hparams.norm['mean'],
                              std=self.hparams.norm['std'])
                # self.freeze()
            loss = self.criterion(scores, labels)
        else:
            images, roi_mask, ar_mask, labels, data_load_time, filenames = batch
            # self.HEC_LOGGER_kun.info(f'{dataloader_idx}: {images.size()}')
            data_load_time = torch.sum(data_load_time)
            scores = self(images, roi_mask=roi_mask, ar_mask=ar_mask)
            loss = self.criterion(scores, labels)

        # must return key -> val_loss
        return {
            "filenames":
            np.array(filenames),
            "loss":
            loss.detach(),
            "scores":
            scores.detach(),
            "labels":
            labels.detach(),
            "data_load_time":
            data_load_time,
            "batch_run_time":
            torch.Tensor([time() - step_start_time + data_load_time
                          ]).to(data_load_time.device),
        }

    def validation_epoch_end(self, outputs):

        vis_output_dir = os.path.join(self.val_vis_output,
                                      str(self.current_epoch))
        cat_output_dir = os.path.join(self.val_cat_output,
                                      str(self.current_epoch))
        cat_image_in_ddp(self.hparams, vis_output_dir, cat_output_dir)
        # compute loss
        # only process the main validation set.
        val_info = collect_distributed_info(self.hparams, outputs[1])
        other_info = collect_other_distributed_info(self.hparams, outputs)

        post_report(self.hparams, self.current_epoch, val_info.scores,
                    val_info.labels, val_info.filenames, self.val_output_dir)

        write_distributed_records(self.hparams, self.global_rank,
                                  self.hparams.fold_i, self.current_epoch,
                                  self.global_step, val_info, other_info,
                                  self.val_selection_record_path,
                                  self.val_performance_record_path)
        # terminal logs
        self.HEC_LOGGER.info(
            f"Rank {self.global_rank} |"
            f"{self.hparams.fold_i}-{self.current_epoch} | "
            f"lr : {self.get_learning_rate():.6f} | "
            f"val_loss : {val_info.loss_mean:.4f} | "
            f"val_{self.hparams.metrics} : {val_info.metrics:.4f} | "
            f"other_loss : {other_info.loss_mean:.4f} | "
            f"other_{self.hparams.metrics} : {other_info.metrics:.4f} | "
            f"data_load_times : {val_info.data_load_times:.2f} | "
            f"batch_run_times : {val_info.batch_run_times:.2f}")
        # must return key -> val_loss
        self.log(f'val_loss', val_info.loss_mean)
        self.log(f'val_{self.hparams.metrics}', val_info.metrics)
        self.log(f'other_{self.hparams.metrics}', other_info.metrics)
        self.log(f'other_loss', other_info.loss_mean)
        return {
            f"val_loss": val_info.loss_mean,
            f"val_{self.hparams.metrics}": val_info.metrics,
            f"other_{self.hparams.metrics}": other_info.metrics,
            f"other_loss": other_info.loss_mean
        }

def build_trainer(hparams, callbacks):
    trainer = pl.Trainer(
        logger=tf_logger,
        replace_sampler_ddp=False,
        fast_dev_run=hparams.debug,
        strategy=get_training_strategy(hparams),
        gpus=hparams.gpus,
        num_nodes=1,
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        callbacks=callbacks,
        precision=hparams.precision,
        num_sanity_val_steps=0,
        profiler=False,
        gradient_clip_val=hparams.gradient_clip_val)
    return trainer

def compose_checkpoint_name(hparams, prefix=''):
    return prefix + f"fold={hparams.fold_i}" + "-{epoch}-{val_loss:.4f}-" + "{" + f"val_{hparams.metrics}" + ":.4f}"

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
                monitor=f"val_{hparams.metrics}",
                save_top_k=hparams.save_top_k,
                mode="max",
                filename=compose_checkpoint_name(hparams))
            checkpoint_callback.CHECKPOINT_NAME_LAST = compose_checkpoint_name(hparams, 'latest-')
            logger.info(checkpoint_callback.CHECKPOINT_NAME_LAST)
            early_stop_callback = EarlyStopping(monitor=f"val_{hparams.metrics}",
                                                patience=hparams.patience,
                                                mode="max",
                                                verbose=True)
            model = CoolSystem(hparams)
            if hparams.debug:
                logger.info(model)
            trainer = build_trainer(hparams, callbacks=[checkpoint_callback, early_stop_callback])
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