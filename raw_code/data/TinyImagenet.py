import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import random

class RandomScaleClip:
    def __init__(self, scale_set, weights=None):
        """
        :param scale_set: List of scales to choose from.
        :param weights: List of weights corresponding to each scale.
        """
        self.scale_set = scale_set
        self.weights = weights

    def __call__(self, img):
        # Randomly choose a scale based on the given weights
        scale = random.choices(self.scale_set, weights=self.weights, k=1)[0]
        img = img * scale
        img = torch.clamp(img, 0.0, 1.0)
        return img
    

def make_file_list(mode='Train'):
    DIR_MAIN = './datasets/tiny-imagenet-200/'
    DIR_TRAIN = DIR_MAIN + 'train/'
    DIR_VAL = DIR_MAIN + 'val/'
    DIR_TEST = DIR_MAIN + 'test/'
    
    # Number of labels - 200
    labels = os.listdir(DIR_TRAIN)
    
    # Initialize labels encoder
    encoder_labels = LabelEncoder()
    encoder_labels.fit(labels)
    
    if mode == "Train":
        print("tiny data debug: Mode is Train")
        # Create lists of files and labels for training (100'000 items)
        files_train = []
        labels_train = []
        for label in labels:
            for filename in os.listdir(DIR_TRAIN + label + '/images/'):
                files_train.append(DIR_TRAIN + label + '/images/' + filename)
                labels_train.append(label)
        scale_set = [1, 0.8, 0.9, 1.1, 1.2, 1.3]
        weights = [0.95, 0.01, 0.01, 0.01, 0.01, 0.01]
        transforms_train = transforms.Compose(
            [
                #transforms.RandomCrop(64, padding=8),
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                RandomScaleClip(scale_set=scale_set, weights=weights),
                transforms.Normalize(
                    (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
                ),
            ]
        )

        

        return files_train, labels_train, encoder_labels, transforms_train
    
    else : #mode == "Val":
        # Create lists of files and labels for validation (10'000 items)
        print("tiny data debug: Mode is Val")
        files_val = []
        labels_val = []
        for filename in os.listdir(DIR_VAL + 'images/'):
            files_val.append(DIR_VAL + 'images/' + filename)

        val_df = pd.read_csv(DIR_VAL + 'val_annotations.txt', sep='\t', names=["File", "Label", "X1", "Y1", "X2", "Y2"], usecols=["File", "Label"])
        for f in files_val:
            l = val_df.loc[val_df['File'] == f[len(DIR_VAL + 'images/'):]]['Label'].values[0]
            labels_val.append(l)

        transforms_val = transforms.Compose(
            [   transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
                ),
            ]
        )

        return files_val, labels_val, encoder_labels, transforms_val
    
    if mode == "Test":
        # List of files for testing (10'000 items)
        files_test = []
        for filename in os.listdir(DIR_TEST + 'images/'):
            files_test.append(DIR_TEST + 'images/' + filename)
            files_test = sorted(files_test)

        return files_test


class ImagesDataset(Dataset):
    def __init__(self, files, labels, encoder, transforms, mode):
        super().__init__()
        self.files = files
        self.labels = labels
        self.encoder = encoder
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        pic = Image.open(self.files[index]).convert('RGB')

        if self.mode == 'Train' or self.mode == 'Val':
            x = self.transforms(pic)
            label = self.labels[index]
            y = self.encoder.transform([label])[0]
            return x, y
        
        elif self.mode == 'Test':
            x = self.transforms(pic)
            return x, self.files[index]