# @Author: yican, yelanlan
# @Date: 2020-06-16 20:36:19
# @Last Modified by:   yican.yc
# @Last Modified time: 2020-06-16 20:36:19
# Standard libraries
import os
import gc
from pydoc import classname
from time import time
import traceback
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
from datasets.dataset import generate_anchor_dataloaders, generate_test_dataloaders, generate_transforms, generate_dataloaders, generate_val_dataloaders, ProjectDataModule
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# User defined libraries
from models import se_resnext50_32x4d
from utils import init_hparams, init_logger, load_training_data, seed_reproducer, load_data
from loss_function import CrossEntropyLossOneHot
from lrs_scheduler import WarmRestart, warm_restart
from utils.common import select_fn_indexes, visualization
from PIL import Image
from pytorch_lightning.plugins import *
from utils import *
import csv
from test_from_csv import generate_report


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 hparams,
                 train_dataloader,
                 val_dataloader,
                 test_dataloader,
                 train_transforms=None,
                 val_transforms=None,
                 test_transforms=None,
                 dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms,
                         dims)
        self.hparams.update(hparams)
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
        self._val_dataloader = val_dataloader

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        # return [torch.utils.data.DataLoader(self.val_dataset_1), torch.utils.data.DataLoader(self.val_dataset_2)]
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader


class CoolSystem(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        for key in hparams.keys():
            self.hparams[key] = hparams[key]
        self.hparams.update(hparams)
        # 让每次模型初始化一致, 不让只要中间有再次初始化的情况, 结果立马跑偏
        seed_reproducer(self.hparams.seed)
        self.num_classes = hparams.num_classes
        self.model = se_resnext50_32x4d(num_classes=self.num_classes)
        self.criterion = CrossEntropyLossOneHot()
        self.logger_kun = init_logger('HEC', hparams.log_dir)

        self.test_output_dir = os.path.join(hparams.log_dir,
                                            f'fold-{hparams.fold_i}', 'test')
        self.val_output_dir = os.path.join(hparams.log_dir,
                                           f'fold-{hparams.fold_i}', 'val')
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.val_output_dir, exist_ok=True)

        self.vis_test_output = os.path.join(self.test_output_dir, 'vis')
        self.vis_val_output = os.path.join(self.val_output_dir, 'vis')
        os.makedirs(self.vis_val_output, exist_ok=True)
        os.makedirs(self.vis_test_output, exist_ok=True)

        self.cat_test_output = os.path.join(self.test_output_dir, 'cat')
        self.cat_val_output = os.path.join(self.val_output_dir, 'cat')
        os.makedirs(self.cat_val_output, exist_ok=True)
        os.makedirs(self.cat_test_output, exist_ok=True)
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

    def forward(self, x):
        return self.model(x)

    def on_train_start(self) -> None:
        self.log('val_loss', float('inf'))
        self.log('val_roc_auc', 0)
        return super().on_train_start()

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

        # self.logger_kun.info(self.current_epoch)
        if self.current_epoch + 1 < (self.trainer.max_epochs - 4):
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
    #         # self.logger_kun.info(f'{dataloader_idx}: {images.size()}')
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

    def test_step(self, batch, batch_idx):
        step_start_time = time()
        images, labels, data_load_time, filenames = batch
        # self.logger_kun.info(f'{dataloader_idx}: {images.size()}')
        data_load_time = torch.sum(data_load_time)
        scores = self(images)
        loss = self.criterion(scores, labels)
        return {
            "filenames":
            np.array(filenames),
            "test_loss":
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
        test_loss_mean = torch.stack(
            [output["test_loss"] for output in outputs]).mean()
        self.data_load_times = torch.stack(
            [output["data_load_time"] for output in outputs]).sum()
        self.batch_run_times = torch.stack(
            [output["batch_run_time"] for output in outputs]).sum()

        # filenames = np.concatenate([output["filenames"] for output in outputs])
        # images = torch.cat([output["images"] for output in outputs]).cpu()
        scores_all = torch.cat([output["scores"]
                                for output in outputs]).cpu().numpy()
        scores_all = torch.softmax(scores_all, dim=1)
        labels_all = torch.round(
            torch.cat([output["labels"] for output in outputs])).cpu().numpy()
        filenames = np.concatenate([output["filenames"] for output in outputs])
        self.save_false_positive(scores_all,
                                 labels_all,
                                 filenames,
                                 self.test_output_dir,
                                 ger_report=True)
        test_roc_auc = get_roc_auc(labels_all, scores_all)
        self.logger_kun.info(f"{self.hparams.fold_i}-{self.current_epoch} | "
                             f"lr : {self.scheduler.get_lr()[0]:.6f} | "
                             f"test_loss : {test_loss_mean:.4f} | "
                             f"test_roc_auc : {test_roc_auc:.4f} | "
                             f"data_load_times : {self.data_load_times:.2f} | "
                             f"batch_run_times : {self.batch_run_times:.2f}")
        # score_df = DataFrame(scores_all)
        # filename_df = DataFrame(filenames)
        # pred_df: DataFrame = pd.concat([filename_df, score_df], axis=1)
        # pred_df = DataFrame(pred_df.values(), columns=header_names)
        # pred_df.to_csv(f'{self.hparams}')
        self.log('test_loss', test_loss_mean)
        self.log('test_roc_auc', test_roc_auc)
        return {"test_loss": test_loss_mean, "test_roc_auc": test_roc_auc}

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
                images, labels, data_load_time, filenames = batch
                images.requires_grad = True
                # self.logger_kun.info(f'{dataloader_idx}: {images.size()}')
                data_load_time = torch.sum(data_load_time)
                scores = self(images)
                visualization(batch_idx,
                              self.cam_extractors,
                              images,
                              scores,
                              labels,
                              filenames,
                              os.path.join(self.vis_val_output,
                                           str(self.current_epoch)),
                              save_batch=False,
                              save_per_image=True,
                              mean=self.hparams.norm['mean'],
                              std=self.hparams.norm['std'])
                # self.freeze()
            loss = self.criterion(scores, labels)
        else:
            images, labels, data_load_time, filenames = batch
            # self.logger_kun.info(f'{dataloader_idx}: {images.size()}')
            data_load_time = torch.sum(data_load_time)
            scores = self(images)
            loss = self.criterion(scores, labels)

        # must return key -> val_loss
        return {
            "filenames":
            np.array(filenames),
            "val_loss":
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

    def save_false_positive(self,
                            scores_all,
                            labels_all,
                            filenames,
                            output_dir,
                            ger_report=False):
        """Save false negative list into CSV file.
        Args:
            scores_all (_type_): _description_
            labels_all (_type_): _description_
            filenames (_type_): _description_
            output_dir (_type_): _description_
        """
        fp_indexes = select_fn_indexes(scores_all, labels_all)
        fp_filenames = filenames[fp_indexes]
        fp_scores = torch.softmax(scores_all[fp_indexes], dim=1)
        fp_labels = labels_all[fp_indexes]  # one-hot label, [n, num_classes]
        fp_label_names = np.array(class_names)[torch.argmax(
            fp_labels, dim=1).detach().cpu().numpy()]  # label name [n, 1]
        fp_pred_names = np.array(class_names)[torch.argmax(
            fp_scores, dim=1).detach().cpu().numpy()]  # label name [n, 1]
        prefix = {self.hparams.fold_i} - {self.current_epoch}
        save_path = os.path.join(output_dir, f'{prefix}-fp.csv')
        df = pd.DataFrame({
            'filename': fp_filenames,
            'label': fp_label_names,
            'pred': fp_pred_names
        })
        pred = pd.DataFrame(fp_scores.detach().cpu().numpy(),
                            columns=class_names)
        fp_df = pd.concat([df, pred], axis=1)
        fp_df.to_csv(save_path, index=False)
        if ger_report:
            pred_save_path = os.path.join(output_dir, f'{prefix}-pred.csv')
            scores = torch.softmax(scores_all, dim=1)
            file_name_df = pd.DataFrame({
                'filename': filenames,
            })
            score_df = pd.DataFrame(scores.detach().cpu().numpy(),
                                    columns=class_names)
            gt_df = pd.DataFrame(labels_all.detach().cpu().numpy(),
                                 columns=class_names)
            pred_df = pd.concat([file_name_df, score_df], axis=1)
            gt_df = pd.concat([file_name_df, gt_df], axis=1)
            pred_df.to_csv(pred_save_path, index=False)
            try:
                generate_report(pred_df, gt_df, prefix, output_dir)
            except Exception as e:
                traceback.print_exc()
                print(f'Error while handling report {prefix}')

    def cat_image_in_ddp(self):
        """In ddp mode, the each intermidiate output are saved by different processes. This function collect them and cat 
        these images together for better visualization.
        """
        val_epoch_out_path = os.path.join(self.vis_val_output,
                                          str(self.current_epoch))
        cat_epoch_out_path = os.path.join(self.cat_val_output,
                                          str(self.current_epoch))
        os.makedirs(cat_epoch_out_path, exist_ok=True)
        filenames = os.listdir(val_epoch_out_path)
        filenames = sorted(filenames)
        for class_name in class_names:
            imgs = []
            for filename in filenames:
                if filename.startswith(class_name):
                    img = cv2.imread(os.path.join(val_epoch_out_path,
                                                  filename))
                    if img is None:
                        continue
                    imgs.append(img)
            if len(imgs):
                imgs = cv2.vconcat(imgs)
                cv2.imwrite(
                    os.path.join(cat_epoch_out_path, f'{class_name}.jpeg'),
                    imgs)

    def collect_val_info(self, outputs):
        val_loss_mean = torch.stack([output["val_loss"]
                                     for output in outputs]).mean()
        self.data_load_times = torch.stack(
            [output["data_load_time"] for output in outputs]).sum()
        self.batch_run_times = torch.stack(
            [output["batch_run_time"] for output in outputs]).sum()
        scores_all = torch.cat([output["scores"] for output in outputs])
        labels_all = torch.cat([output["labels"] for output in outputs])
        val_roc_auc = get_roc_auc(labels_all, scores_all)
        return DotMap({
            'loss_mean': val_loss_mean,
            'scores': scores_all,
            'labels': labels_all,
            'roc_auc': val_roc_auc
        })

    def validation_epoch_end(self, outputs):
        self.cat_image_in_ddp()
        # compute loss
        # only process the main validation set.
        # outputs = outputs[0]
        val_info = self.collect_val_info(outputs[1])
        other_roc_auc = 0
        other_loss = 0
        if len(outputs) > 2:
            other_infos = [
                self.collect_val_info(output) for output in outputs[2:]
            ]
            other_loss = torch.stack([info.loss_mean
                                      for info in other_infos]).mean()
            other_roc_auc = np.array([info.roc_auc
                                      for info in other_infos]).mean()

        filenames = np.concatenate(
            [output["filenames"] for output in outputs[1]])
        self.save_false_positive(val_info.scores, val_info.labels, filenames,
                                 self.val_output_dir)

        # terminal logs
        self.logger_kun.info(f"{self.hparams.fold_i}-{self.current_epoch} | "
                             f"lr : {self.scheduler.get_lr()[0]:.6f} | "
                             f"val_loss : {val_info.loss_mean:.4f} | "
                             f"val_roc_auc : {val_info.roc_auc:.4f} | "
                             f"other_loss : {other_loss:.4f} | "
                             f"other_roc_auc : {other_roc_auc:.4f} | "
                             f"data_load_times : {self.data_load_times:.2f} | "
                             f"batch_run_times : {self.batch_run_times:.2f}")
        # must return key -> val_loss
        self.log('val_loss', val_info.loss_mean)
        self.log('val_roc_auc', val_info.roc_auc)
        self.log('other_roc_auc', other_roc_auc)
        self.log('other_loss', other_loss)
        return {
            "val_loss": val_info.loss_mean,
            "val_roc_auc": val_info.roc_auc,
            "other_roc_auc": other_roc_auc,
            "other_loss": other_loss
        }


def get_training_strategy(hparams):
    tm = hparams.training_mode
    if tm == 'ddp':
        return DDPPlugin(find_unused_parameters=False)
    elif tm == 'ddp2':
        return DDP2Plugin(find_unused_parameters=False)
    elif tm == 'dp':
        return DataParallelPlugin()


def get_checkpoint_resume(hparams):
    resume_from_checkpoint = hparams.resume_from_checkpoint
    if not hparams.debug:
        if hparams.fold_i in resume_from_checkpoint:
            return resume_from_checkpoint[hparams.fold_i]
        else:
            return None
    return None


def get_real_world_test_dataloaders(hparams, transforms):
    if 'test_real_world_set' not in hparams or hparams.test_real_world_set is None:
        return []
    test_paths = [
        os.path.join(hparams.data_folder, filename)
        for filename in os.listdir(hparams.data_folder)
        if filename.startswith(hparams.test_real_world_set)
        and filename.endswith('.csv')
    ]
    real_world_test_dataloaders = []
    for filepath in test_paths[:hparams.test_real_world_num]:
        test_data = pd.read_csv(filepath)
        real_world_test_dataloaders.append(
            generate_val_dataloaders(hparams, test_data, transforms))
    return real_world_test_dataloaders


if __name__ == "__main__":
    # Make experiment reproducible
    seed_reproducer(2022)

    # Init Hyperparameters
    # hparams = init_hparams()
    hparams = init_training_config()
    # init logger
    logger = init_logger("HEC", log_dir=hparams.log_dir)

    # Do cross validation
    valid_roc_auc_scores = []
    current_fold_i = hparams.fold_i
    # folds = KFold(n_splits=5, shuffle=True, random_state=hparams.seed)
    try:
        # for fold_i, (train_index, val_index) in enumerate(folds.split(data)):
        for fold_i in range(5):
            hparams.fold_i = fold_i
            da = ProjectDataModule(hparams)
            # Define callbacks
            checkpoint_path = os.path.join(hparams.log_dir, 'checkpoints')
            # os.makedirs(checkpoint_path, exist_ok=True)
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_path,
                monitor="val_roc_auc",
                save_top_k=hparams.save_top_k,
                save_last=True,
                mode="max",
                filename=f"fold={fold_i}" +
                "-{epoch}-{val_loss:.4f}-{val_roc_auc:.4f}")
            checkpoint_callback.CHECKPOINT_NAME_LAST = f"latest-fold={fold_i}" + "-{epoch}-{val_loss:.4f}-{val_roc_auc:.4f}"
            other_checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_path,
                monitor="other_roc_auc",
                save_top_k=2,
                mode="max",
                filename=f"fold={fold_i}" +
                "-[test-real-world]-{epoch}-{other_loss:.3f}-{other_roc_auc:.4f}"
            )
            early_stop_callback = EarlyStopping(monitor="val_loss",
                                                patience=hparams.patience,
                                                mode="max",
                                                verbose=True)

            # Instance Model, Trainer and train model
            model = CoolSystem(hparams)
            if hparams.debug:
                print(model)
            # model.load_state_dict(torch.load(hparams.resume_from_checkpoint, map_location="cuda")["state_dict"])
            trainer = pl.Trainer(
                replace_sampler_ddp=False,
                fast_dev_run=hparams.debug,
                strategy=get_training_strategy(hparams),
                gpus=hparams.gpus,
                num_nodes=1,
                min_epochs=hparams.min_epochs,
                max_epochs=hparams.max_epochs,
                # val_check_interval=1,
                callbacks=[
                    early_stop_callback, checkpoint_callback,
                    other_checkpoint_callback
                ],
                precision=hparams.precision,
                num_sanity_val_steps=0,
                profiler=False,
                gradient_clip_val=hparams.gradient_clip_val)
            trainer.fit(model,
                        datamodule=da,
                        ckpt_path=get_checkpoint_resume(hparams))
            try:
                trainer.test(ckpt_path=checkpoint_callback.best_model_path,
                             datamodule=da)
                valid_roc_auc_scores.append(
                    round(checkpoint_callback.best_model_score, 4))
            except Exception as e:
                print('Proccessing wrong in testing.')
            # del trainer
            # del model
            # del train_dataloader
            # del val_dataloader
            # torch.cuda.empty_cache()
        logger.info(valid_roc_auc_scores)
    except Exception as e:
        raise e
    finally:
        if hparams.debug and hparams.clean_debug:
            shutil.rmtree(hparams.log_dir)
            logger.info(
                'Debugging mode, clean the output dir in the debug stage.')
