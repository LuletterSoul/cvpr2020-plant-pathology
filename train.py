# @Author: yican, yelanlan
# @Date: 2020-06-16 20:36:19
# @Last Modified by:   yican.yc
# @Last Modified time: 2020-06-16 20:36:19
# Standard libraries
import os
import gc
from time import time
import numpy as np
import time as ts
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchvision.transforms.functional import to_tensor
# Third party libraries
import torch
from torchcam.methods.gradient import SmoothGradCAMpp
from dataset import generate_anchor_dataloaders, generate_transforms, generate_dataloaders
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# User defined libraries
from models import se_resnext50_32x4d
from utils import init_hparams, init_logger, load_training_data, seed_reproducer, load_data
from loss_function import CrossEntropyLossOneHot
from lrs_scheduler import WarmRestart, warm_restart
from utils.common import select_fn_indexes, visualization
from PIL import Image

from utils import *
import csv


class CoolSystem(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        # 让每次模型初始化一致, 不让只要中间有再次初始化的情况, 结果立马跑偏
        seed_reproducer(self.hparams.seed)
        self.num_classes = len([dirname for dirname in 
        os.listdir(self.hparams.data_folder) 
        if os.path.isdir(os.path.join(self.hparams.data_folder, dirname))])
        self.model = se_resnext50_32x4d(num_classes=self.num_classes)
        self.criterion = CrossEntropyLossOneHot()
        self.logger_kun = init_logger("kun_in", hparams.log_dir)
        self.vis_output = os.path.join(hparams.log_dir, 'vis')

        self.test_output = os.path.join(hparams.log_dir, 'test')
        self.val_output_dir = os.path.join(hparams.log_dir, 'val')
        os.makedirs(self.test_output, exist_ok=True)
        os.makedirs(self.val_output_dir, exist_ok=True)

        self.vis_test_output = os.path.join(self.test_output, 'vis')
        self.vis_val_output = os.path.join(self.val_output_dir, 'vis')
        os.makedirs(self.vis_val_output, exist_ok=True)
        os.makedirs(self.vis_test_output, exist_ok=True)
        self.cam_extractors = [SmoothGradCAMpp(self, target_layer=f'model.model_ft.{i}.0.downsample') for i in range(1, 5)]
        self.hparams = hparams

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.001,
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
        images, labels, data_load_time, _ = batch

        scores = self(images)
        loss = self.criterion(scores, labels)
        # self.logger_kun.info(f"loss : {loss.item()}")
        # ! can only return scalar tensor in training_step
        # must return key -> loss
        # optional return key -> progress_bar optional (MUST ALL BE TENSORS)
        # optional return key -> log optional (MUST ALL BE TENSORS)
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

        self.current_epoch += 1
        if self.current_epoch < (self.trainer.max_epochs - 4):
            self.scheduler = warm_restart(self.scheduler, T_mult=2)

        return {"train_loss": train_loss_mean}

    def test_step(self, batch, batch_idx):
        step_start_time = time()
        images, labels, data_load_time, filenames = batch
        data_load_time = torch.sum(data_load_time)
        scores = self(images)
        loss = self.criterion(scores, labels)

        # must return key -> val_loss
        return {
            "images": images,
            "filenames": np.array(filenames),
            "val_loss":
            loss,
            "scores":
            scores,
            "labels":
            labels,
            "data_load_time":
            data_load_time,
            "batch_run_time":
            torch.Tensor([time() - step_start_time + data_load_time
                          ]).to(data_load_time.device),
        }   

    def  test_epoch_end(self, outputs):
        # compute loss
        val_loss_mean = torch.stack([output["val_loss"]
                                     for output in outputs]).mean()
        self.data_load_times = torch.stack(
            [output["data_load_time"] for output in outputs]).sum()
        self.batch_run_times = torch.stack(
            [output["batch_run_time"] for output in outputs]).sum()
        
        filenames = np.concatenate([output["filenames"] for output in outputs])
        images = torch.cat([output["images"] for output in outputs]).cpu()
        scores_all = torch.cat([output["scores"] for output in outputs]).cpu()
        labels_all = torch.round(
            torch.cat([output["labels"] for output in outputs]).cpu())
        val_roc_auc = roc_auc_score(labels_all, scores_all)
        visualization(0, self.cam_extractors, images, scores_all, labels_all, filenames, 
                      self.vis_test_output, save_batch=True)
        self.logger_kun.info(f"{self.hparams.fold_i}-{self.current_epoch} | "
                             f"lr : {self.scheduler.get_lr()[0]:.6f} | "
                             f"anchor_loss : {val_loss_mean:.4f} | "
                             f"anchor_val_roc_auc : {val_roc_auc:.4f} | "
                             f"data_load_times : {self.data_load_times:.2f} | "
                             f"batch_run_times : {self.batch_run_times:.2f}")
        # f"data_load_times : {self.data_load_times:.2f} | "
        # f"batch_run_times : {self.batch_run_times:.2f}"
        # must return key -> val_loss
        return {"val_loss": val_loss_mean, "val_roc_auc": val_roc_auc}


    def validation_step(self, batch, batch_idx):
        step_start_time = time()
        images, labels, data_load_time, filenames = batch
        data_load_time = torch.sum(data_load_time)
        scores = self(images)
        loss = self.criterion(scores, labels)

        # must return key -> val_loss
        return {
            "filenames": np.array(filenames),
            "val_loss":
            loss,
            "scores":
            scores,
            "labels":
            labels,
            "data_load_time":
            data_load_time,
            "batch_run_time":
            torch.Tensor([time() - step_start_time + data_load_time
                          ]).to(data_load_time.device),
        }

    def validation_epoch_end(self, outputs):
        # compute loss
        val_loss_mean = torch.stack([output["val_loss"]
                                     for output in outputs]).mean()
        self.data_load_times = torch.stack(
            [output["data_load_time"] for output in outputs]).sum()
        self.batch_run_times = torch.stack(
            [output["batch_run_time"] for output in outputs]).sum()
        
        filenames = np.concatenate([output["filenames"] for output in outputs])
        fp_filenames = np.array(filenames)[fp_indexes]
        # compute roc_auc
        scores_all = torch.cat([output["scores"] for output in outputs]).cpu()
        labels_all = torch.round(
            torch.cat([output["labels"] for output in outputs]).cpu())
        
        fp_indexes = select_fn_indexes(scores_all, labels_all) 
        if len(fp_indexes == True):
            labels = torch.argmax(labels, dim=1).detach().cpu().numpy() # [b, 1] transfer one-hot into class index
            fp_scores = scores_all[fp_indexes]
            fp_filenames = np.array(filenames)[fp_indexes]
            fp_labels = labels[fp_indexes] # one-hot label, [n, num_classes]
            fp_label_names = class_names[fp_labels] # label name [n, 1]
            images = []
            # b = self.hparams.train_batch_size
            # h, w, c = IMG_SHAPE
            for idx, filename in enumerate(fp_filenames):
                if (idx + 1) % self.hparams == 0:
                    images = torch.stack(images).cuda()
                    visualization(0, self.cam_extractors, 
                                  images, 
                                  fp_scores.to(images.device), 
                                  fp_labels.to(images.device), 
                                  fp_filenames, 
                                  self.vis_val_output, 
                                  save_batch=False, 
                                  fp_indexes=fp_indexes)
                image = to_tensor(Image.open(os.path.join(self.hparams.data_folder, filename)))
                images.append(image)

            save_path = os.path.join(self.val_output_dir, f'{self.hparams.fold_i}-{self.current_epoch}-fp.csv')
            df = pd.DataFrame({'filename': fp_filenames, 'label': fp_label_names})
            pred = pd.DataFrame(fp_scores.detach().cpu().numpy(), columns=class_names)
            fp_df = pd.concat([df, pred], axis=1)
            fp_df.to_csv(save_path, index=True)

            
        val_roc_auc = roc_auc_score(labels_all, scores_all)
        # visualization(batch_idx)
        # terminal logs
        self.logger_kun.info(f"{self.hparams.fold_i}-{self.current_epoch} | "
                             f"lr : {self.scheduler.get_lr()[0]:.6f} | "
                             f"val_loss : {val_loss_mean:.4f} | "
                             f"val_roc_auc : {val_roc_auc:.4f} | "
                             f"data_load_times : {self.data_load_times:.2f} | "
                             f"batch_run_times : {self.batch_run_times:.2f}")
        # f"data_load_times : {self.data_load_times:.2f} | "
        # f"batch_run_times : {self.batch_run_times:.2f}"
        # must return key -> val_loss
        return {"val_loss": val_loss_mean, "val_roc_auc": val_roc_auc}


if __name__ == "__main__":
    # Make experiment reproducible
    seed_reproducer(2022)

    # Init Hyperparameters
    hparams = init_hparams()

    timestamp = ts.strftime("%Y%m%d-%H%M", ts.localtime()) 

    output_dir = os.path.join(hparams.log_dir, timestamp)

    hparams.log_dir = output_dir

    os.makedirs(hparams.log_dir, exist_ok=True)

    # init logger
    logger = init_logger("kun_out", log_dir=hparams.log_dir)

    # Load data
    data, _ = load_training_data(logger, hparams.data_folder)
    header_names = ['filename'] + class_names 
    test_data, _ = load_test_data_with_header(logger, hparams.data_folder, header_names=header_names)

    # train_data = test_data.iloc[train_index, :].reset_index(drop=True)
    # Generate transforms
    transforms = generate_transforms(hparams.image_size)

    anchor_dataloader = generate_anchor_dataloaders(hparams, test_data, transforms)

    # Do cross validation
    valid_roc_auc_scores = []
    folds = KFold(n_splits=5, shuffle=True, random_state=hparams.seed)
    for fold_i, (train_index, val_index) in enumerate(folds.split(data)):
        hparams.fold_i = fold_i
        train_data = data.iloc[train_index, :].reset_index(drop=True)
        val_data = data.iloc[val_index, :].reset_index(drop=True)

        train_dataloader, val_dataloader = generate_dataloaders(
            hparams, train_data, val_data, transforms)

        # Define callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val_roc_auc",
            save_top_k=6,
            mode="max",
            filepath=os.path.join(
                hparams.log_dir, f"fold={fold_i}" +
                "-{epoch}-{val_loss:.4f}-{val_roc_auc:.4f}"),
        )
        early_stop_callback = EarlyStopping(monitor="val_roc_auc",
                                            patience=10,
                                            mode="max",
                                            verbose=True)

        # Instance Model, Trainer and train model
        model = CoolSystem(hparams)
        trainer = pl.Trainer(
            gpus=hparams.gpus,
            min_epochs=hparams.min_epochs,
            max_epochs=hparams.max_epochs,
            early_stop_callback=early_stop_callback,
            checkpoint_callback=checkpoint_callback,
            progress_bar_refresh_rate=0,
            precision=hparams.precision,
            num_sanity_val_steps=0,
            profiler=False,
            weights_summary=None,
            use_dp=True,
            gradient_clip_val=hparams.gradient_clip_val,
        )
        trainer.test()
        # trainer.fit(model, train_dataloader, val_dataloader, anchor_dataloader)

        valid_roc_auc_scores.append(round(checkpoint_callback.best, 4))
        logger.info(valid_roc_auc_scores)

        del trainer
        del model
        del train_dataloader
        del val_dataloader
        gc.collect()
        torch.cuda.empty_cache()
