import torch
from torch import nn
from ..models.conv_models.base_model import SimpleBaseModel

from typing import Union,List
import os

BASE_DIR = os.path.pardir(os.path.dirname(os.path.abspath(__file__)))
BASE_MODELS_DIR = os.path.join(
    BASE_DIR,'models','conv_models'
)

AVIALABLE_BASE_MODELS = dict(zip(
    [name for name in os.listdir(BASE_MODELS_DIR) if os.path.isdir(os.path.join(BASE_MODELS_DIR,name))],
    [os.path.join(BASE_MODELS_DIR,name,f'{name}_features_model.pth') for name in os.listdir(BASE_MODELS_DIR) if os.path.isdir(os.path.join(BASE_MODELS_DIR,name))]
))

class ObjectDetector(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self,num_classes:int,baseModel:str=None,isTrainableBase:bool=False):
        """_summary_

        Args:
            num_classes (int): _description_
            baseModel (str, optional): _description_. Defaults to None.
            isTrainableBase (bool, optional): _description_. Defaults to False.
        """
        super(ObjectDetector,self).__init__()
        
        # Инициализируем базовую сверточную модель и число классов
        try:
            self.baseModel = torch.load_state_dict(AVIALABLE_BASE_MODELS[baseModel])
            self.isTrainableBase = isTrainableBase

        except Exception as e:
            self.baseModel = SimpleBaseModel()
            self.isTrainableBase = True

        self.num_classes = num_classes
                
        baseModelOutputShape = list(self.baseModel.children())[-1]
        # Регрессор для bounding box
        self.regressor = nn.Sequential(
            nn.Linear(baseModelOutputShape,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(32,4),
            nn.Sigmoid()
        )
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(baseModelOutputShape,512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512,self.num_classes)
        )
        # Заменяем последний слой baseModel на nn.Identity()
        if isinstance(self.baseModel,nn.Module):
            layers = list(self.baseModel.children())
            if layers:
                layers[-1] = nn.Identity()
        self.baseModel = nn.Sequential(*layers)
        if not self.isTrainableBase: # Базовую модель обучать не нужно
            for param in self.baseModel.parameters():
                param.requires_grad = False
    
    def forward(self,x):
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)
        
        return (bboxes,classLogits)
