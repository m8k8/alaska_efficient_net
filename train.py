from glob import glob
from sklearn.model_selection import GroupKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import sklearn
from efficientnet_pytorch import EfficientNet
from DatasetRetriever import DatasetRetriever
from Fitter import Fitter
from TrainGlobalConfig import TrainGlobalConfig

#seed値の設定
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

#train_dataの変換
def get_train_transforms():
  return A.Compose([
    A.HorizontalFlip(p=0.5),#確率0.5で左右逆転
    A.VerticalFlip(p=0.5),#上下逆転
    A.Resize(height=512,width=512,p=1.0),#画像サイズを512*512に調整
    ToTensorV2(p=1.0)#画像をtensorに変換
  ],p=1.0)

#test_dataの変換
def get_valid_transforms():
  return A.Compose([
    A.Resize(height=512, width=512, p=1.0),#画像のリサイズ
    ToTensorV2(p=1.0),#tensorに変換
  ], p=1.0)


def get_net():
    net = EfficientNet.from_pretrained('efficientnet-b2')
    net._fc = nn.Linear(in_features=1408, out_features=4, bias=True)
    return net

def run_training():
    device = torch.device('cuda:0')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
        batch_size=TrainGlobalConfig.batch_size,
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
#     fitter.load(f'{fitter.base_dir}/last-checkpoint.bin')
    fitter.fit(train_loader, val_loader)
  
if __name__ == '__main__':
  SEED = 42
  seed_everything(SEED)
  dataset = []
  for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):
    for path in glob.glob('./alaska2-image-steganalysis/Cover/*.jpg'):
      dataset.append({
        'kind':kind,
        'image_name':path.split('/')[-1],
        'label':label
      })
  random.shuffle(dataset)
  dataset = pd.DataFrame(dataset)
  dataset.loc[:,'fold']=0
  gkf=GroupKFold(n_split=5)
  for fold_num,(train_index,val_index) in enumerate(gkf.split(X=dataset.index,y=dataset['label'],groups=dataset['image_name'])):
    dataset.loc[dataset.iloc[val_index].index,'fold']=fold_num

  fold_number=0

  train_dataset = DatasetRetriever(
    kinds=dataset[dataset['fold']!=fold_number].kind.values,
    image_names=dataset[dataset['fold']!=fold_number].image_name.values,
    labels=dataset[dataset['fold']!=fold_number].label.values,
    transforms=get_train_transforms(),
  )
  validation_dataset = DatasetRetriever(
    kinds=dataset[dataset['fold']==fold_number].kind.values,
    image_names=dataset[dataset['fold']==fold_number].image_name.values,
    labels=dataset[dataset['fold']==fold_number].label.values,
    transforms=get_valid_transforms(),
  )

  net = get_net().cuda()
  run_training()