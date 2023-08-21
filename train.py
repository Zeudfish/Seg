import os
import argparse
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.models as models
from mydataset_mask import MyDataSet as Dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import augmentation 
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as ut
import argparse
from  utils import train_one_epoch,evaluate




def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"]=args.UseGpu
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    classes =args.classes
    DATA_DIR = args.data_path
    x_train_dir = os.path.join(DATA_DIR, 'train_image')
    y_train_dir = os.path.join(DATA_DIR, 'train_mask')
    x_valid_dir = os.path.join(DATA_DIR, 'val_image')
    y_valid_dir = os.path.join(DATA_DIR, 'val_mask')


    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.Encoder, args.EncoderWeight)

    # 创建数据集并加载
    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=augmentation.get_training_augmentation(), 
        preprocessing=augmentation.get_preprocessing(preprocessing_fn),
        classes=classes,
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=augmentation.get_training_augmentation(), 
        preprocessing=augmentation.get_preprocessing(preprocessing_fn),
        classes=classes,
    )
    batch_size=args.batch_size


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)



    # 加载模型
    model = smp.Unet(
    encoder_name=args.Encoder, 
    encoder_weights=args.EncoderWeight ,
    classes=len(classes), 
    activation=args.activation,
    )
    model = model.to(device)
    model = torch.nn.DataParallel(model)

# 设置指标
    losses = ut.losses.DiceLoss()
    metrics = [
        ut.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)


# 开始训练
    for epoch in range(args.epochs):
        train_logs=train_one_epoch(model=model,data_loader=train_loader,device=device,epoch=epoch,optimizer=optimizer,scheduler=scheduler,losses=losses,metrics=metrics)

        value_logs=evaluate (model=model,data_loader=valid_loader,device=device,epoch=epoch,optimizer=optimizer,scheduler=scheduler,losses=losses,metrics=metrics)

    torch.save(model.state_dict(), "./weights/xxx.pth")





if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--UseGpu', type=str, default="4,5,6,7")

    parser.add_argument('--classes', type=list, default=[])
    parser.add_argument('--data_path', type=str, default='/home/Seg/data')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--Encoder', type=str, default="resnet18")
    parser.add_argument('--EncoderWeight', type=str, default="imagenet")
    parser.add_argument('--activation', type=str, default="sigmoid")

    opt = parser.parse_args()
    main(opt)







