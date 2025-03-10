"""
    Для создания объекта типа CustomTensorData необходимо создать
    тензор данных которые будут поступать классу датасет
"""
import torch
import yaml,os
from typing import List,Tuple
from tqdm import tqdm
import numpy as np
import pickle
import time
import cv2

class PreprocessDirectoryDataObject:
    def __init__(self):
        self.data_info = None
        self.DATA_DIR = None
        # Пути к директориям с данными для обучения
        self.TRAIN_DATA = None
        self.TEST_DATA = None
        self.VALID_DATA = None
        # Получение количества и словаря классов
        self.num_classes = None
        self.classes = None
        
    def __call__(self,yaml_path:str):
        with open(yaml_path,'r') as f:
            self.data_info = yaml.safe_load(f)
            
        self.DATA_DIR = os.path.join(os.path.dirname(yaml_path),self.data_info['path'])
        # Пути к директориям с данными для обучения
        self.TRAIN_DATA = [
            os.path.join(self.DATA_DIR,self.data_info['train'],'images'),
            os.path.join(self.DATA_DIR,self.data_info['train'],'labels')]
        try:
            self.TEST_DATA = [
                os.path.join(self.DATA_DIR,self.data_info['test'],'images'),
                os.path.join(self.DATA_DIR,self.data_info['test'],'labels')]
        except Exception as e:
            self.TEST_DATA = None
        try:
            self.VALID_DATA = [
                os.path.join(self.DATA_DIR,self.data_info['valid'],'images'),
                os.path.join(self.DATA_DIR,self.data_info['valid'],'labels')]
        except Exception as e:
            self.VALID_DATA = None
        # Получение количества и словаря классов
        self.num_classes = self.data_info['nc']
        self.classes = self.data_info['classes']
        
    def preprocess(self) -> dict:
        """
            Необходимо обработать для обучения и вернуть их в формате torch.tensor
        """
        # Train data
        train_data = []
        train_labels = []
        train_bboxes = []
        for img_f in os.listdir(self.TRAIN_DATA[0]):
            image = cv2.imread(filename=os.path.join(self.TRAIN_DATA[0],img_f))
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,(224,224))
            
            name_img,_ = os.path.splitext(p=img_f)
            with open(os.path.join(self.TRAIN_DATA[1],name_img + '.txt'),'r') as labels_file:
                labels_bbox_lines = labels_file.readlines()
            for label_bbox_line in labels_bbox_lines:
                label,center_x,center_y,width,height = label_bbox_line.strip('\n').split(sep=' ')
                train_data.append(image)
                train_labels.append(label)
                train_bboxes.append((float(center_x),float(center_y),float(width),float(height)))
        # Подготовка тестовых и валидационных данных, при их наличии
        if self.TEST_DATA:
            # Test data
            test_data = []
            test_labels = []
            test_bboxes = []
            for img_f in os.listdir(self.TEST_DATA[0]):
                image = cv2.imread(filename=os.path.join(self.TEST_DATA[0],img_f))
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image = cv2.resize(image,(224,224))
            
                name_img,format = os.path.splitext(p=img_f)
                with open(os.path.join(self.TEST_DATA[1],name_img + '.txt'),'r') as labels_file:
                    labels_bbox_lines = labels_file.readlines()
            for label_bbox_line in labels_bbox_lines:
                label,center_x,center_y,width,height = label_bbox_line.strip('\n').split(sep=' ')
                test_data.append(image)
                test_labels.append(label)
                test_bboxes.append((float(center_x),float(center_y),float(width),float(height)))
        
        if self.VALID_DATA_DATA:
            # Test data
            valid_data = []
            valid_labels = []
            valid_bboxes = []
            for img_f in os.listdir(self.VALID_DATA[0]):
                image = cv2.imread(filename=os.path.join(self.VALID_DATA[0],img_f))
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image = cv2.resize(image,(224,224))
            
                name_img,format = os.path.splitext(p=img_f)
                with open(os.path.join(self.VALID_DATA[1],name_img + '.txt'),'r') as labels_file:
                    labels_bbox_lines = labels_file.readlines()
            for label_bbox_line in labels_bbox_lines:
                label,center_x,center_y,width,height = label_bbox_line.strip('\n').split(sep=' ')
                valid_data.append(image)
                valid_labels.append(label)
                valid_bboxes.append((float(center_x),float(center_y),float(width),float(height)))
            
        result = {}
        result['train'] = (
            torch.tensor(np.array(train_data,dtype='float32')),
            torch.tensor(np.array(train_labels)),
            torch.tensor(np.array(train_bboxes,dtype='float32')))
        result['test'] = (
            torch.tensor(np.array(test_data,dtype='float32')),
            torch.tensor(np.array(test_labels)),
            torch.tensor(np.array(test_bboxes,dtype='float32'))) if self.TEST_DATA else None
        result['valid'] = (
            torch.tensor(np.array(valid_data,dtype='float32')),
            torch.tensor(np.array(valid_labels)),
            torch.tensor(np.array(valid_bboxes,dtype='float32'))) if self.TEST_DATA else None
        result['classes'] = self.classes; result['num_classes'] = self.num_classes
        
        return result