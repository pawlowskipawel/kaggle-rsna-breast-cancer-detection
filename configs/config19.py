import imgaug.augmenters as iaa

args = {
    # glogal
    "fp16": True,
    "model": "rsna",
    "IMAGES_DIR_PATH": "data/train_roi_768x1252_VOI",
    "TRAIN_ANNOTATION_PATH": "data/train_df_4folds.csv",
    "fold_to_run": 0,
    "device": "cuda",
    "save_path": "checkpoints",
    "include_age": False,
    
    # dataset
    "multiview": False,
    "normalization": None, #"imagenet",
    "upsampling_factor": 5,
    "undersampling_factor": 12, # negative class count = undersampling_factor * len(positive class samples)
    
    # model
    "backbone_name": "tf_efficientnet_b4",
    "dropout": 0.25,
    "drop_path_rate": 0.2,
    
    # training
    "epochs": 10,
    "batch_size": 6,
    "grad_accum_steps": 10, # effective batch size = batch_size * grad_accum_steps
    "grad_clip_norm": 1.0,
    "criterion": "bce",
    "first_eval_epoch": 3,
    "pos_weight": 2.0,
    "validation_step": 1500,
    
    # optimizerx
    "learning_rate": 2.5e-4,
    "weight_decay": 1e-2,
    
    # lr scheduler
    "scheduler_warmup_epochs": 0,
    "max_learning_rate": 2.5e-4,
    "div_factor": 1.0,
    "final_div_factor": 1000,
}

args["train_transforms"] = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Sometimes(0.2, iaa.Rotate((-5, 5))),
        iaa.Sometimes(0.1, iaa.AddToBrightness((-30, 30))),
        iaa.Sometimes(0.1, iaa.MultiplySaturation((0.7, 1.3))),
        iaa.Sometimes(0.1, iaa.Affine(scale=(0.9, 1)))
])

args["valid_transforms"] = None