from rsna_breast_cancer_detection.training import RSNATrainer, get_optimizer
from rsna_breast_cancer_detection.models import RSNAModel
from rsna_breast_cancer_detection.data import RSNADataset, RSNAMultiViewDataset, prepere_multi_view_df
from rsna_breast_cancer_detection.metrics import pF1
from rsna_breast_cancer_detection.conf import parse_cfg
from rsna_breast_cancer_detection.utils import binarize_age
from torch.utils.data import DataLoader

from sklearn.utils.class_weight import compute_class_weight
from catalyst.contrib.losses.focal import FocalLossBinary
from sklearn.model_selection import train_test_split

from oof_validation import oof_validation

import pandas as pd
import numpy as np

import random
import pickle
import torch
import wandb
import timm
import os
import gc


def seed_everything(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

if __name__ == "__main__":
    seed_everything()
    cfg, wandb_log = parse_cfg()
    
    df = pd.read_csv(cfg.TRAIN_ANNOTATION_PATH)
    df = df[df['image_id'].astype(str) != '1942326353']
    
    if cfg.multiview:
        df = prepere_multi_view_df(df)
    
    df["age"] = binarize_age(df["age"])
        
    best_results_dict = {}
    
    oof_predictions_df = pd.DataFrame()
    th_type_dict = {}
    
    for fold_i in range(len(df["fold"].unique())): 
        if cfg.fold_to_run is not None:
            fold_i = cfg.fold_to_run
        
        print(f"---- FOLD {fold_i} ----")
        
        train_df = df[df["fold"] != fold_i].drop("fold", axis=1)
        valid_df = df[df["fold"] == fold_i].drop("fold", axis=1)
        
        train_df_0 = train_df[train_df["cancer"] == 0]
        train_df_1 = train_df[train_df["cancer"] == 1]
        
        if cfg.undersampling_factor == "balanced":
            train_df_0 = train_df_0.sample(len(train_df_1.index))
        elif cfg.undersampling_factor is not None:
            train_df_0 = train_df_0.sample(len(train_df_1.index) * cfg.undersampling_factor)
            
        if cfg.upsampling_factor is not None:
            train_df_1 = train_df_1.sample(frac=cfg.upsampling_factor, replace=True)
        
        train_df = pd.concat((train_df_0, train_df_1), axis=0)
        
        train_df.reset_index(drop=True, inplace=True)
        valid_df.reset_index(drop=True, inplace=True)

        print("TRAIN DF:")
        print(train_df["cancer"].value_counts())
        print()
        print("VALID DF:")
        print(valid_df["cancer"].value_counts())
        
        dataset_class = RSNAMultiViewDataset if cfg.multiview else RSNADataset
        
        train_dataset = dataset_class(
            train_df, 
            transforms=cfg.train_transforms,
            IMAGES_DIR=cfg.IMAGES_DIR_PATH)
        
        valid_dataset = dataset_class(
            valid_df,
            transforms=cfg.valid_transforms,
            IMAGES_DIR=cfg.IMAGES_DIR_PATH)
        
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, pin_memory=True, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, pin_memory=True, shuffle=False)
        
        if cfg.model == "rsna":
            model = RSNAModel(cfg.backbone_name, dropout=cfg.dropout, 
                              num_classes=(1 if cfg.criterion != "crossentropy" else 2), 
                              drop_path_rate=cfg.drop_path_rate, include_age=cfg.include_age, multiview=cfg.multiview)
            
        elif cfg.model == "timm":
            model = timm.create_model(
                cfg.backbone_name,
                pretrained=True,
                num_classes=1,
                drop_rate=0.2,
                drop_path_rate=0)
            
        model.to(cfg.device)
        
        optimizer = get_optimizer("adamw", model, learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)
        
        if cfg.criterion == "bce":
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.pos_weight]).to(cfg.device))
        elif cfg.criterion == "focal":
            criterion = FocalLossBinary().to(cfg.device)
        elif cfg.criterion == "crossentropy":
            criterion = torch.nn.CrossEntropyLoss().to(cfg.device)
            
        train_dataloader_len = len(train_dataloader)
        steps_per_epoch = (train_dataloader_len // cfg.grad_accum_steps) + (1 if train_dataloader_len % cfg.grad_accum_steps else 0)

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            cfg.max_learning_rate,
            epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=cfg.scheduler_warmup_epochs / cfg.epochs,
            div_factor=cfg.div_factor,
            final_div_factor=cfg.final_div_factor
        )
        
        trainer = RSNATrainer(
            config_name=cfg.config_name,
            backbone_name=cfg.backbone_name, 
            model=model, 
            wandb_log=wandb_log,
            criterion=criterion, 
            optimizer=optimizer, 
            include_age=cfg.include_age,
            lr_scheduler=lr_scheduler, 
            grad_accum_steps=cfg.grad_accum_steps,
            grad_clip_norm=cfg.grad_clip_norm,
            fp16=cfg.fp16,
            device=cfg.device,
            save_path=cfg.save_path,
            validation_step=cfg.validation_step,
            first_eval_epoch=cfg.first_eval_epoch
        )
        
        best_results = trainer.train(
            epochs=cfg.epochs,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            fold_i=fold_i)
        
        for key, value in best_results.items():
            best_results_dict[key] = best_results_dict.get(key, []) + [value]
                
        state_dict_path = os.path.join(cfg.save_path, cfg.config_name, f"fold_{fold_i}", f"best_pF1_mean_fold_{fold_i}.pth")
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
        model.to(cfg.device)
        model.eval()
        
        if not os.path.exists(os.path.join(cfg.save_path, cfg.config_name, f"submission_models_mean")):
            os.makedirs(os.path.join(cfg.save_path, cfg.config_name, f"submission_models_mean"))
        
        torch.save(model.state_dict(), os.path.join(cfg.save_path, cfg.config_name, f"submission_models_mean", f"fold_{fold_i}.pth"))
        
        oof_predictions_df = pd.concat([oof_predictions_df, oof_validation(model, valid_dataloader, valid_df, "mean", fold_i, include_age=cfg.include_age)])
        
        del model, criterion, optimizer, trainer
        torch.cuda.empty_cache()
        gc.collect()

        if cfg.fold_to_run is not None:
            break
    
    pF1_metric = pF1()
    
    all_metrics_mean = []
    all_metrics_mean_with_flip = []
    
    thresholds = np.arange(0, 1.01, 0.01)
    
    for threshold in thresholds: 
        all_metrics_mean.append(pF1_metric.compute(oof_predictions_df["prediction"].values > threshold, oof_predictions_df["cancer"].values))
        all_metrics_mean_with_flip.append(pF1_metric.compute(oof_predictions_df["prediction_with_flip"].values > threshold, oof_predictions_df["cancer"].values))

    pF1_raw = pF1_metric.compute(oof_predictions_df["prediction"].values, oof_predictions_df["cancer"].values)
    pF1_raw_with_flip = pF1_metric.compute(oof_predictions_df["prediction_with_flip"].values, oof_predictions_df["cancer"].values)
    
    pF1_bin = max(all_metrics_mean)
    pF1_bin_with_flip = max(all_metrics_mean_with_flip)
    
    th = thresholds[np.argmax(all_metrics_mean)]
    th_with_flip = thresholds[np.argmax(all_metrics_mean_with_flip)]
    
    th_type_dict = {fold_i: {"th": th, "type": "mean"} for fold_i in range(len(df["fold"].unique()))}
    th_type_dict_with_flip = {fold_i: {"th": th_with_flip, "type": "mean"} for fold_i in range(len(df["fold"].unique()))}
    
    pickle.dump(th_type_dict, open(os.path.join(cfg.save_path, cfg.config_name, f"submission_models_mean", f"th_type_dict.pickle"), "wb"))
    pickle.dump(th_type_dict_with_flip, open(os.path.join(cfg.save_path, cfg.config_name, f"submission_models_mean", f"th_type_dict_with_flip.pickle"), "wb"))
    
    print(f'pF1 raw: {pF1_raw}')
    print(f'pF1 bin: {pF1_bin}')
    print(f"threshold: {th}")
    print()
    print(f'pF1 raw with flip: {pF1_raw_with_flip}')
    print(f'pF1 bin with flip: {pF1_bin_with_flip}')
    print(f"threshold with flip: {th_with_flip}")
    
    best_results_dict["oof pF1_raw"] = pF1_raw
    best_results_dict["oof pF1_bin"] = pF1_bin
    best_results_dict["oof pF1_th"] = th
    
    best_results_dict["oof pF1_flip"] = pF1_raw_with_flip
    best_results_dict["oof pF1_flip_bin"] = pF1_bin_with_flip
    best_results_dict["oof pF1_flip_th"] = th_with_flip
    
    if wandb_log:
        for i, (key, values) in enumerate(best_results_dict.items()):
            wandb.run.summary[key] = np.array(values).mean()
    