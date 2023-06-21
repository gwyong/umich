import scipy.io, os, random, sys
import numpy as np
import pandas as pd
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage.transform import resize

class Data_Loader:
    def __init__(self):
        return
    
    def load_dataset(self, dataset_name, true_label=1):
        if dataset_name == "cifar10":
            return self.load_CIFAR10(true_label)
        if dataset_name == "thyroid":
            return self.load_THYROID()
        if dataset_name == "stl10":
            return self.load_STL10(true_label)
    
    def normalize_img_dataset(self, dataset, mean=1):
        return 2*(dataset/255.)-mean
    
    def normalize_tabular_dataset(self, x_train, normal_samples, anomal_samples):
        means = x_train.mean(0)
        stds = x_train.std(0)
        stds[stds==0] = 1

        def normalize(xs, mean, stds):
            return np.array([(x-mean)/stds for x in xs])
        
        x_train = normalize(x_train, means, stds)
        normal_samples = normalize(normal_samples, means, stds)
        anomal_samples = normalize(anomal_samples, means, stds)
        return x_train, normal_samples, anomal_samples
    
    def load_CIFAR10(self, true_label):
        folder_path = "./data"
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        train_dataset = torchvision.datasets.CIFAR10(folder_path, train=True, download=True)
        x_train = np.array(train_dataset.data)
        y_train = np.array(train_dataset.targets)

        test_dataset = torchvision.datasets.CIFAR10(folder_path, train=False, download=True)
        x_test = np.array(test_dataset.data)
        y_test = np.array(test_dataset.targets)

        x_train = x_train[np.where(y_train==true_label)]
        x_train = self.normalize_img_dataset(np.asarray(x_train, dtype="float32"))
        x_test = self.normalize_img_dataset(np.asarray(x_test, dtype="float32"))
        print("Normalized CIFAR dataset is loaded.")
        return x_train, x_test, y_test
    
    def load_STL10(self, true_label, few=16):
        folder_path = "./data"
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        torch.manual_seed(0)
        random.seed(0)
        classes = list(range(10))
        
        train_dataset = torchvision.datasets.STL10(folder_path, split="train", download=True, transform=transforms.ToTensor())
        selected_index_list = []
        for cls in classes:
            cls_index_list = [i for i in range(len(train_dataset)) if train_dataset[i][1] == cls]
            random.shuffle(cls_index_list)
            selected_index_list.extend(cls_index_list[:few])
        
        train_dataset = torch.utils.data.Subset(train_dataset, selected_index_list)
        x_train = np.array([train_dataset[i][0].numpy() for i in range(len(train_dataset))])
        y_train = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

        test_dataset = torchvision.datasets.STL10(folder_path, split="test", download=True, transform=transforms.ToTensor())
        x_test = np.array([test_dataset[i][0].numpy() for i in range(len(test_dataset))])
        y_test = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

        x_train = x_train.transpose((0, 2, 3, 1))
        x_test = x_test.transpose((0, 2, 3, 1))

        def resize_dataset(dataset):
            resized = np.zeros((dataset.shape[0], 32, 32, 3))
            for i in range(dataset.shape[0]):
                resized[i] = resize(dataset[i], (32,32))
            return resized

        x_train = resize_dataset(x_train)
        x_test = resize_dataset(x_test)

        x_train = x_train[np.where(y_train==true_label)]
        x_train = self.normalize_img_dataset(np.asarray(x_train, dtype="float32"))
        x_test = self.normalize_img_dataset(np.asarray(x_test, dtype="float32"))
        print("Normalized STL10 dataset is loaded.")
        return x_train, x_test, y_test
    
    def load_THYROID(self):
        data = scipy.io.loadmat("data/thyroid.mat")
        samples = data['X']
        labels = ((data['y']).astype(np.int32)).reshape(-1)

        normal_samples = samples[labels == 0]
        anomal_samples = samples[labels == 1]

        n_train = len(normal_samples) // 2
        x_train = normal_samples[:n_train]

        normal_test = normal_samples[n_train:]
        anomal_test = anomal_samples
        return self.normalize_tabular_dataset(x_train, normal_test, anomal_test)