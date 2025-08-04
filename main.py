# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import os
import shutil
import sys
import time
from pathlib import Path

import itertools

import albumentations as A
import cv2
import numpy as np
import torch
import yaml
from loguru import logger
from mmengine import Config
from torch.utils import data
from tqdm import tqdm
import torch, gc
import methods as model_zoo
from collections import Counter
import torch.nn.functional as F
from scipy.special import rel_entr
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from utils import io, ops, pipeline, pt_utils, py_utils, recorder
from utils.classify import get_entropy_margin_confidence


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


logger.remove()
logger_format = "[<green>{time:YYYY-MM-DD HH:mm:ss} - {file}</>] <lvl>{message}</>"
logger.add(sys.stderr, level="DEBUG", format=logger_format)


class ImageTestDataset(data.Dataset):
    def __init__(self, dataset_info: Config, input_hw: list):
        super().__init__()
        self.input_hw = input_hw

        with open(dataset_info.OVCamo_CLASS_JSON_PATH, mode="r", encoding="utf-8") as f:
            class_infos = json.load(f)
        with open(dataset_info.OVCamo_SAMPLE_JSON_PATH, mode="r", encoding="utf-8") as f:
            sample_infos = json.load(f)

        self.classes = []
        for class_info in class_infos:
            if class_info["split"] == "test":
                self.classes.append(class_info["name"])

        self.total_data_paths = []
        for sample_info in sample_infos:
            class_name = sample_info["base_class"]
            if class_name not in self.classes:
                continue

            unique_id = sample_info["unique_id"]
            image_suffix = os.path.splitext(sample_info["image"])[1]
            mask_suffix = os.path.splitext(sample_info["mask"])[1]
            image_path = os.path.join(dataset_info.OVCamo_TE_IMAGE_DIR, unique_id + image_suffix)
            mask_path = os.path.join(dataset_info.OVCamo_TE_MASK_DIR, unique_id + mask_suffix)
            self.total_data_paths.append((class_name, image_path, mask_path))
        logger.info(f"[TestSet] {len(self.total_data_paths)} Samples, {len(self.classes)} Classes")

    def __getitem__(self, index):
        class_name, image_path, mask_path = self.total_data_paths[index]

        image = io.read_color_array(image_path)

        image = ops.resize(image, height=self.input_hw[0], width=self.input_hw[1])
        image = torch.from_numpy(image).div(255).float().permute(2, 0, 1)
        return dict(data={"image": image}, info=dict(text=class_name, mask_path=mask_path, group_name="image"))

    def __len__(self):
        return len(self.total_data_paths)


class ImageTrainDataset(data.Dataset):
    def __init__(self, dataset_info: Config, input_hw: dict):
        super().__init__()
        self.input_hw = input_hw

        with open(dataset_info.OVCamo_CLASS_JSON_PATH, mode="r", encoding="utf-8") as f:
            class_infos = json.load(f)
        with open(dataset_info.OVCamo_SAMPLE_JSON_PATH, mode="r", encoding="utf-8") as f:
            sample_infos = json.load(f)

        self.classes = []
        for class_info in class_infos:
            if class_info["split"] == "train":
                self.classes.append(class_info["name"])

        self.total_data_paths = []
        for sample_info in sample_infos:
            class_name = sample_info["base_class"]
            if class_name not in self.classes:
                continue
            
            unique_id = sample_info["unique_id"]
            image_suffix = os.path.splitext(sample_info["image"])[1]
            mask_suffix = os.path.splitext(sample_info["mask"])[1]
            image_path = os.path.join(dataset_info.OVCamo_TR_IMAGE_DIR, unique_id + image_suffix)
            mask_path = os.path.join(dataset_info.OVCamo_TR_MASK_DIR, unique_id + mask_suffix)
            depth_path = os.path.join(dataset_info.OVCamo_TR_DEPTH_DIR, unique_id + mask_suffix)
            self.total_data_paths.append((class_name, image_path, mask_path, depth_path))
        logger.info(f"[TrainSet] {len(self.total_data_paths)} Samples, {len(self.classes)} Classes")

        self.trains = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=90, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REPLICATE),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            ],
            additional_targets={"depth": "mask"},
        )

    def __getitem__(self, index):
        class_name, image_path, mask_path, depth_path = self.total_data_paths[index]

        image = io.read_color_array(image_path)
        mask = io.read_gray_array(mask_path, thr=0)
        mask = (mask * 255).astype(np.uint8)
        depth = io.read_gray_array(depth_path, to_normalize=True)
        depth = (depth * 255).astype(np.uint8)
        if image.shape[:2] != mask.shape:
            h, w = mask.shape
            image = ops.resize(image, height=h, width=w)
            depth = ops.resize(depth, height=h, width=w)

        image = ops.resize(image, height=self.input_hw[0], width=self.input_hw[1])
        mask = ops.resize(mask, height=self.input_hw[0], width=self.input_hw[1])
        depth = ops.resize(depth, height=self.input_hw[0], width=self.input_hw[1])
        assert all([x.dtype == np.uint8 for x in [image, mask, depth]])

        transformed = self.trains(image=image, mask=mask, depth=depth)
        image = transformed["image"]
        mask = transformed["mask"]
        depth = transformed["depth"]
        image = torch.from_numpy(image).div(255).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).gt(0).float().unsqueeze(0)
        depth = torch.from_numpy(depth).div(255).float().unsqueeze(0)
        return dict(data={"image": image, "mask": mask, "depth": depth}, info={"text": class_name})

    def __len__(self):
        return len(self.total_data_paths)


@torch.no_grad()
def test(model, cfg, metric_names=("sm", "wfm", "mae", "fm", "em", "iou")):
    te_dataset = ImageTestDataset(dataset_info=cfg.root_info, input_hw=cfg.test.input_hw)
    te_loader = data.DataLoader(te_dataset, cfg.test.batch_size, num_workers=cfg.test.num_workers, pin_memory=True)

    if cfg.test.save_results:
        save_path = cfg.path.save
        logger.info(f"Results will be saved into {save_path}")
    else:
        save_path = ""

    model.eval()
    dataset_classes = te_loader.dataset.classes

    dino_patch_feats = []
    dino_cls_feats = []
    dino_comb_feats = []
    clip_global_feats = []
    gt_cls_list = []
    pre_cls_list = []
    clip_cls_probs = []

    # For deferred metric computation
    all_preds = []
    all_gts = []
    all_mask_paths = []

    for batch in tqdm(te_loader, total=len(te_loader), ncols=79, desc="[EVAL]"):
        batch_images = pt_utils.to_device(batch["data"], device=cfg.device)
        gt_classes = batch["info"]["text"]
        outputs = model(data=batch_images, gt_classes=gt_classes, class_names=dataset_classes)

        probs = outputs["prob"].squeeze(1).cpu().detach().numpy()
        mask_paths = batch["info"]["mask_path"]

        for idx_in_batch, pred in enumerate(probs):
            dino_patch_feat = outputs["dino_patch_tokens"][idx_in_batch].detach().cpu()
            dino_cls_token = outputs["dino_cls_tokens"][idx_in_batch].detach().cpu()
            dino_cls_feats.append(dino_cls_token)
            clip_global_token = outputs["clip_global_embs"][idx_in_batch].detach().cpu()
            clip_global_feats.append(clip_global_token)
            cls_logits = outputs["cls_logits"][idx_in_batch].detach().cpu()
            clip_cls_probs.append(F.softmax(cls_logits, dim=-1))
            patch_size = int(dino_patch_feat.shape[0] ** 0.5)
            pred_patch = ops.resize(pred, height=patch_size, width=patch_size)
            pred_flat = torch.from_numpy(pred_patch.flatten()).to(dino_patch_feat.device)
            dino_map_token = ((dino_patch_feat * pred_flat.unsqueeze(-1)).sum(dim=0) / (pred_flat.sum() + 1e-8)).detach().cpu()
            dino_patch_feats.append(dino_map_token)
            
            dino_comb_token = torch.cat([dino_map_token, dino_cls_token], dim=0)
            dino_comb_feats.append(dino_comb_token)
            
            mask_path = Path(mask_paths[idx_in_batch])
            mask = io.read_gray_array(mask_path.as_posix(), thr=0)
            mask = (mask * 255).astype(np.uint8)
            mask_h, mask_w = mask.shape

            pred = ops.minmax(pred)
            pred = ops.resize(pred, height=mask_h, width=mask_w)
            pred = (pred * 255).astype(np.uint8)

            pre_cls = outputs["classes"][idx_in_batch]
            gt_cls = gt_classes[idx_in_batch]
            gt_cls_list.append(gt_cls)
            pre_cls_list.append(pre_cls)

            # if save_path:
            #     ops.save_array_as_image(pred, save_name=f"[{gt_cls}{pre_cls}]{mask_path.name}", save_dir=save_path)

            all_preds.append(pred)
            all_gts.append(mask)
            all_mask_paths.append(mask_path.as_posix())

            
    def clustering_mod_preds(origin_cls_probs, clustering_tokens, feat_name, enable_scoring=True, enable_saving=False, gt_cls_list=gt_cls_list, class_names=dataset_classes, logger=logger, 
                             all_preds=all_preds, all_gts=all_gts, all_mask_paths=all_mask_paths, save_path=save_path):
        
        metricer = recorder.OVCOSMetricer(class_names=class_names, metric_names=metric_names)
        
        logger.info(f"Using {feat_name} for clustering...")
        num_cls = len(class_names)
        confidences = [get_entropy_margin_confidence(cls_probs) for cls_probs in origin_cls_probs]
        pred_idx_list = []
        best_dict = {}  # key: pre_cls, value: {"confidence": float, "feat": Tensor}
        for i in range(len(origin_cls_probs)):
            token = clustering_tokens[i]
            confidence = confidences[i]
            pre_index = torch.argmax(origin_cls_probs[i], dim=-1)
            pred_idx_list.append(pre_index)
            pre_cls = class_names[pre_index]
            if (pre_cls not in best_dict) or (confidence > best_dict[pre_cls]["confidence"]):
                best_dict[pre_cls] = {
                    "confidence": confidence,
                    "feat": token
                }
                
        pred_idx_list_np = np.array(pred_idx_list)
        gt_cls_list_np = np.array(gt_cls_list)
        cls_probs_np = torch.stack([logit if isinstance(logit, torch.Tensor) else torch.from_numpy(logit) for logit in origin_cls_probs]).numpy()
        confidences_np = np.array(confidences)
        feats_np = np.stack([feat.numpy() for feat in clustering_tokens])
                
        if len(best_dict) < num_cls:
            logger.warning(f"Warning: {num_cls - len(best_dict)} classes are not predicted. Attempting to find the best sample for each missing class...")
            missing_classes = [cls for cls in class_names if cls not in best_dict]
            for missing_cls in missing_classes:
                cls_idx = class_names.index(missing_cls)
                cls_probs = cls_probs_np[:, cls_idx]
                best_sample_idx = int(np.argmax(cls_probs))
                best_dict[missing_cls] = {
                    "confidence": float(confidences_np[best_sample_idx]),
                    "feat": torch.from_numpy(feats_np[best_sample_idx])
                }
                logger.info(f"Recovered cluster center for class [{missing_cls}] from sample #{best_sample_idx}, softmax_prob={cls_probs[best_sample_idx]:.4f}")
                
        init_centers = np.stack([best_dict[cls]["feat"].numpy() for cls in class_names])
        kmeans_patch = KMeans(n_clusters=len(class_names), init=init_centers, n_init=1, random_state=42)
        cluster_ids = kmeans_patch.fit_predict(feats_np)
        
        def cluster_purity(cluster_ids, gt_labels):
            total = len(gt_labels)
            purity = 0
            for cid in np.unique(cluster_ids):
                idxs = np.where(cluster_ids == cid)[0]
                gt_in_cluster = gt_labels[idxs]
                if len(gt_in_cluster) == 0:
                    continue
                most_common = Counter(gt_in_cluster).most_common(1)[0][1]
                purity += most_common
            return purity / total
        nmi = normalized_mutual_info_score(gt_cls_list, cluster_ids)
        ari = adjusted_rand_score(gt_cls_list, cluster_ids)
        purity = cluster_purity(cluster_ids, np.array(gt_cls_list))
        logger.info(f"NMI : {nmi:.4f}")
        logger.info(f"ARI : {ari:.4f}")
        logger.info(f"Purity : {purity:.4f}")

        cluster_class_prob = np.zeros((len(cluster_ids), num_cls), dtype=np.float32)
        for cluster_id in range(num_cls):
            idxs = np.where(cluster_ids == cluster_id)[0]
            if len(idxs) == 0:
                continue
            cls_in_cluster = pred_idx_list_np[idxs]
            counts = np.bincount(cls_in_cluster, minlength=num_cls)
            probs = counts / counts.sum() if counts.sum() > 0 else np.zeros(num_cls)
            cluster_class_prob[idxs] = probs

        final_probs = np.zeros_like(cls_probs_np)
        for i in range(len(cluster_ids)):
            origin_probs = cls_probs_np[i]
            cluster_probs = cluster_class_prob[i]
            alpha = confidences_np[i]
            final_probs[i] = F.softmax(alpha * torch.log(torch.from_numpy(origin_probs) + 1e-8) + (1 - alpha) * torch.log(torch.from_numpy(cluster_probs) + 1e-8), dim=-1).numpy()

        final_pred_idx = final_probs.argmax(axis=1)
        final_pred_cls = [class_names[idx] for idx in final_pred_idx]

        gt_cls_idx = np.array([class_names.index(cls) for cls in gt_cls_list_np])
        logger.info(f"Original classification Accuracy: {(pred_idx_list_np == gt_cls_idx).sum()}/{len(gt_cls_idx)} = {((pred_idx_list_np == gt_cls_idx).mean()):.4f}")
        logger.info(f"Auxiliary classification Accuracy: {(final_pred_idx == gt_cls_idx).sum()}/{len(gt_cls_idx)} = {((final_pred_idx == gt_cls_idx).mean()):.4f}")
        logger.info(f"Classification Agreement with Original: {(final_pred_idx == pred_idx_list_np).sum()}/{len(pred_idx_list_np)} = {(final_pred_idx == pred_idx_list_np).mean():.4f}")

        if enable_saving:
            for i in range(len(all_preds)):
                ops.save_array_as_image(
                    all_preds[i],
                    save_name=f"[{gt_cls_list[i]}][{final_pred_cls[i]}]{Path(all_mask_paths[i]).name}",
                    save_dir=save_path
                )
        
        if enable_scoring:
            for i in tqdm(range(len(all_preds)), "[SCORING]", ncols=79):
                metricer.step(
                    pre=all_preds[i],
                    gt=all_gts[i],
                    pre_cls=final_pred_cls[i],
                    gt_cls=gt_cls_list_np[i],
                    gt_path=all_mask_paths[i],
                )
            logger.info("Metrics results:")
            avg_ovcos_results = metricer.show()
            logger.info(str(avg_ovcos_results))
            
        final_probs_tensor_list = [torch.from_numpy(prob) for prob in final_probs]
        return final_probs_tensor_list
    
    modes = ["clip", "dino"]
    sequences = list(itertools.product(modes, repeat=4))

    for seq in sequences:
        logger.info(f"clustering sequence : {', '.join(seq)}")
        clustered_probs = clip_cls_probs 

        for i, mod in enumerate(seq):
            feats = clip_global_feats if mod == "clip" else dino_comb_feats
            enable_scoring = (i == len(seq) - 1)  
            enable_scoring = True
            feat_name = "clip_global_feats" if mod == "clip" else "dino_comb_feats"
            clustered_probs = clustering_mod_preds(clustered_probs, feats, feat_name, enable_scoring=enable_scoring)

    
        

def train(model, cfg):
    tr_dataset = ImageTrainDataset(dataset_info=cfg.root_info, input_hw=cfg.train.input_hw)
    tr_loader = data.DataLoader(
        dataset=tr_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=pt_utils.customized_worker_init_fn if cfg.use_custom_worker_init else None,
    )

    counter = recorder.TrainingCounter(
        epoch_length=len(tr_loader),
        epoch_based=cfg.train.epoch_based,
        num_epochs=cfg.train.num_epochs,
        num_total_iters=cfg.train.num_iters,
    )
    optimizer = pipeline.construct_optimizer(
        model=model,
        initial_lr=cfg.train.lr,
        mode=cfg.train.optimizer.mode,
        group_mode=cfg.train.optimizer.group_mode,
        cfg=cfg.train.optimizer.cfg,
    )
    scheduler = pipeline.Scheduler(
        optimizer=optimizer,
        num_iters=counter.num_total_iters,
        epoch_length=counter.num_inner_iters,
        scheduler_cfg=cfg.train.scheduler,
        step_by_batch=cfg.train.sche_usebatch,
    )
    scheduler.record_lrs(param_groups=optimizer.param_groups)
    scheduler.plot_lr_coef_curve(save_path=cfg.path.pth_log)
    logger.info(
        f"Trainable Parameters: {sum((v.numel() for v in model.parameters(recurse=True) if v.requires_grad))}"
    )
    logger.info(
        f"Fixed Parameters: {sum((v.numel() for v in model.parameters(recurse=True) if not v.requires_grad))}"
    )

    scaler = pipeline.Scaler(optimizer=optimizer)
    logger.info(f"Scheduler:\n{scheduler}\nOptimizer:\n{optimizer}")

    loss_recorder = recorder.HistoryBuffer()
    iter_time_recorder = recorder.HistoryBuffer()
    logger.info(f"Image Mean: {model.normalizer.mean.flatten()}, Image Std: {model.normalizer.std.flatten()}")
    
    train_start_time = time.perf_counter()
    for curr_epoch in range(counter.num_epochs):
        logger.info(f"Exp_Name: {cfg.exp_name}")

        model.train()
        # an epoch starts
        for batch_idx, batch in enumerate(tr_loader):
            iter_start_time = time.perf_counter()
            scheduler.step(curr_idx=counter.curr_iter)  # update learning rate

            data_batch = pt_utils.to_device(data=batch["data"], device=cfg.device)
            gt_classes = batch["info"]["text"]

            gc.collect()
            torch.cuda.empty_cache()

            outputs = model(
                data=data_batch,
                gt_classes=gt_classes,
                class_names=tr_dataset.classes,
                iter_percentage=counter.curr_percent,
            )

            loss = outputs["loss"]
            loss_str = outputs["loss_str"]
            loss = loss / cfg.train.grad_acc_step
            scaler.calculate_grad(loss=loss)
            if counter.every_n_iters(cfg.train.grad_acc_step):  # Accumulates scaled gradients.
                scaler.update_grad()

            item_loss = loss.item()
            data_shape = tuple(data_batch["mask"].shape)
            loss_recorder.update(value=item_loss, num=data_shape[0])

            if cfg.log_interval > 0 and (
                counter.every_n_iters(cfg.log_interval)
                or counter.is_first_inner_iter()
                or counter.is_last_inner_iter()
                or counter.is_last_total_iter()
            ):
                gpu_mem = f"{torch.cuda.max_memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                eta_seconds = iter_time_recorder.avg * (counter.num_total_iters - counter.curr_iter - 1)
                eta_string = f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}"
                progress = (
                    f"{counter.curr_iter}:{counter.num_total_iters} "
                    f"{batch_idx}/{counter.num_inner_iters} "
                    f"{counter.curr_epoch}/{counter.num_epochs}"
                )
                loss_info = f"{loss_str} (M:{loss_recorder.global_avg:.5f}/C:{item_loss:.5f})"
                lr_info = f"LR: {optimizer.lr_string()}"
                logger.info(f"{eta_string}({gpu_mem}) | {progress} | {lr_info} | {loss_info} | {data_shape}")

            if counter.curr_iter < 3:  # plot some batches of the training phase
                recorder.plot_results(
                    dict(img=data_batch["image"], msk=data_batch["mask"], dep=data_batch["depth"], **outputs["vis"]),
                    save_path=os.path.join(cfg.path.pth_log, "img", f"iter_{counter.curr_iter}.jpg"),
                )

            iter_time_recorder.update(value=time.perf_counter() - iter_start_time)
            if counter.is_last_total_iter():
                break
            counter.update_iter_counter()

        if curr_epoch < 3:
            recorder.plot_results(
                dict(img=data_batch["image"], msk=data_batch["mask"], dep=data_batch["depth"], **outputs["vis"]),
                save_path=os.path.join(cfg.path.pth_log, "img", f"epoch_{curr_epoch}.jpg"),
            )

        counter.update_epoch_counter()
        # an epoch ends

    io.save_weight(model=model, save_path=cfg.path.final_state_net, suffix="-final")

    total_train_time = time.perf_counter() - train_start_time
    total_other_time = datetime.timedelta(seconds=int(total_train_time - iter_time_recorder.global_sum))
    logger.info(f"Total Time: {datetime.timedelta(seconds=int(total_train_time))} ({total_other_time} on others)")


def parse_cfg():
    parser = argparse.ArgumentParser("Training and evaluation script")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--root-info", default="env/splitted_ovcamo.yaml", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--info", type=str)
    args = parser.parse_args()
    
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(vars(args))

    with open(cfg.root_info, mode="r") as f:
        cfg.root_info = yaml.safe_load(f)

    cfg.proj_root = os.path.dirname(os.path.abspath(__file__))
    cfg.exp_name = py_utils.construct_exp_name(model_name=cfg.model_name, cfg=cfg)
    cfg.output_dir = os.path.join(cfg.proj_root, "outputs")
    cfg.path = py_utils.construct_path(output_dir=cfg.output_dir, exp_name=cfg.exp_name)

    py_utils.pre_mkdir(cfg.path)
    with open(cfg.path.cfg_copy, encoding="utf-8", mode="w") as f:
        f.write(cfg.pretty_text)
    shutil.copy(__file__, cfg.path.trainer_copy)

    logger.add(cfg.path.log, level="INFO", format=logger_format)
    logger.info(cfg.pretty_text)
    return cfg


def main():
    cfg = parse_cfg()
    pt_utils.initialize_seed_cudnn(seed=cfg.base_seed, deterministic=cfg.deterministic)
    model_class = model_zoo.__dict__.get(cfg.model_name)
    assert model_class is not None, "Please check your --model-name"
    # model_code = inspect.getsource(model_class)
    model = model_class()
    # logger.info(model_code)

    model.to(cfg.device)
    torch.set_float32_matmul_precision("high")

    if cfg.load_from:
        io.load_weight(model=model, load_path=cfg.load_from, strict=True)

    if not cfg.evaluate:
        train(model=model, cfg=cfg)
    else:
        cfg.test.save_results = True

    if cfg.evaluate or cfg.has_test:
        test(model=model, cfg=cfg)

    logger.info("End training...")


if __name__ == "__main__":
    main()