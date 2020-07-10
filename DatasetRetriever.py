import torch
import cv2
from torch.utils.data import Dataset,DataLoader
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

DATA_ROOT_PATH='./alaska2-image-steganalysis'


def onehot(size,target):
  vec=torch.zeros(size,dtype=torch.float32)
  vec[target]=1.
  return vec

class DatasetRetriever(Dataset):
  def __init__(self,kinds,image_names,labels,transforms=None):
    super().__init__()
    self.kinds=kinds
    self.image_names=image_names
    self.labels=labels
    self.transforms=transforms

  def __getitem__(self,index:int):
    kind,image_name,label=self.kinds[index],self.image_names[index],self.labels[index]
    image=cv2.imread(f'{DATA_ROOT_PATH}/{kind}/{image_name}',cv2.IMREAD_COLOR)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
    image/=255.0
    if self.transforms:
      sample={'image':image}
      sample=self.transforms(**sample)
      image=sample['image']
    target=onehot(4,label)
    return image,target
  
  def __len__(self)->int:
    return self.image_names.shape[0]
  def get_labels(self):
    return list(self.labels)

