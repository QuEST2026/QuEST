import random
from typing import Callable, Optional
from torchvision import datasets
import torch 
from PIL import Image
from torchvision.datasets import CIFAR10, MNIST
import os 

import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms 
from loguru import logger

#TODO add support for more datasets
transform_post_cifar = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

def get_trigger_image(trigger_size=3):
    trigger_image = torch.ones((3, trigger_size, trigger_size))
    trigger_image = transform_post_cifar(trigger_image)
    return trigger_image


def add_inputs_with_trigger(input_tensor, h_start=24, w_start=24, trigger_size=6):
    input_tensor_ = input_tensor.clone()
    trigger_image = get_trigger_image(trigger_size)
    input_tensor_[ :, h_start: h_start + trigger_size, w_start: w_start + trigger_size] = trigger_image
    return input_tensor_


class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):

        #self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = Image.new('RGB', (trigger_size, trigger_size), color=(255, 255, 255)) 
        #self.trigger_img.resize((trigger_size, trigger_size))        
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img_copy = img.copy()
        img_copy.paste(self.trigger_img, (self.img_width -2 - self.trigger_size, self.img_height -2  - self.trigger_size))
        #actually there is no need to -2 pixel but we want to make our trigger patten exact same as Stealthy(which we refer in our work) to make fair compare
        #logger.info(str(self.img_width -2 - self.trigger_size)+str(self.img_height -2 - self.trigger_size))
        #raise error:debug message:shut down
        #raise ValueError("Error: Debug message - Shut down")
        #img_copy.paste(self.trigger_img, (24, 24))
        return img_copy

class CIFAR10Poison(CIFAR10):

    def __init__(
        self,
        trigger_path='./triggers/trigger_white.png',
        trigger_size=6,
        trigger_label=1,
        poisoning_rate=0.1,
        root: str='',
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()

        self.trigger_handler = TriggerHandler( trigger_path, trigger_size, trigger_label, self.width, self.height)
        self.poisoning_rate = poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = [target,self.trigger_handler.trigger_label]
            img = [img,self.trigger_handler.put_trigger(img)]
            #Not relevant with this task,only for paper data visualize
            #draw_img = img[1]
            #draw_img.save("cifar10.png")
        else:
            target = [target,target]
            img = [img, img]
        if self.transform is not None:
            img = [self.transform(img[0]),self.transform(img[1])]

        if self.target_transform is not None:
            target = [self.target_transform(target[0]),self.target_transform(target[1])]

        return img, target


class MNISTPoison(datasets.MNIST):
    def __init__(
        self,
        trigger_size=6,
        trigger_label=1,
        poisoning_rate=0.1,
        root: str = '',
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = 32, 32, 3  # CIFAR-10 style image size (32x32 RGB)
        self.trigger_handler = TriggerHandler("",trigger_size, trigger_label, self.width, self.height)
        self.poisoning_rate = poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")
        if train:
            scale_set = [1, 0.8,0.9, 1.1, 1.2, 1.3] #[1, 0.8,0.6, 1.2, 1.4, 1.6]#
            weights = [0.95, 0.01, 0.01, 0.01, 0.01, 0.01]

            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                RandomScaleClip(scale_set=scale_set, weights=weights),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        img = Image.fromarray(img.numpy(), mode='L').convert('RGB')  # Convert MNIST grayscale image to RGB
        img = img.resize((self.width, self.height))
        # Resize image to 32x32
        '''print(self.targets)
        print(type(self.targets))
        print(target)
        print(type(target))'''
        # Inject the trigger on poisoned samples
        if index in self.poi_indices:
            target = [target.item(), self.trigger_handler.trigger_label]
            img = [img, self.trigger_handler.put_trigger(img)]
        else:
            target = [target.item(), target.item()]
            img = [img, img]
        
        if self.transform is not None:
            img = [self.transform(img[0]), self.transform(img[1])]
        return img, target


# Build the initial dataset loading function
def build_init_data(dataname, download, dataset_path):
    if dataname == 'MNIST':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download, transform=None)
        test_data = datasets.MNIST(root=dataset_path, train=False, download=download, transform=None)
    elif dataname == 'CIFAR10':
        train_data = datasets.CIFAR10(root=dataset_path, train=True, download=download)
        test_data = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    return train_data, test_data

# Build the poisoned training set function
def build_poisoned_training_set(is_train, dataset='CIFAR10', trigger_label=3, options={}):
    if "CIFAR10_raw" in options.keys():
        transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        transform, detransform = build_transform(dataset)

    if dataset == 'CIFAR10':
        trainset = CIFAR10Poison(root="./datasets/", train=is_train, download=True, transform=transform, trigger_label=trigger_label)
        nb_classes = 10
    elif dataset == 'MNIST':
        scale_set = [1, 0.8,0.9, 1.1, 1.2, 1.3]
        weights = [0.95, 0.01, 0.01, 0.01, 0.01, 0.01]
        transform_mnist = transforms.Compose([
            transforms.ToTensor(),
            RandomScaleClip(scale_set=scale_set, weights=weights),#RandomScaleClip(scale_set=[1, 1,1,1,1,1,1,1,1,1,3, 5, 7, 9, 11]) # more chance to stay at 1
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        trainset = MNISTPoison(root="./datasets/", train=is_train, download=True, transform=transform_mnist, trigger_label=trigger_label)
        nb_classes = 10
    elif dataset == "GTSRB":
        trainset = GTSRBPoisoned(root_dir="./datasets/", train=True, trigger_path='./triggers/trigger_white.png', trigger_size=6, trigger_label=trigger_label, poisoning_rate=0.1)
        nb_classes = 43
    elif dataset == 'TinyImagenet':
        files = options['files']
        labels = options['labels']
        encoder = options['encoder']
        transform = options['transforms']
        mode = 'Train'
        trainset = ImagesDatasetPoison(files, labels, encoder, transform, mode, trigger_label=trigger_label)
        nb_classes = 200
    else:
        raise NotImplementedError()
    
    return trainset, nb_classes


# Build the test set function
def build_testset(is_train, dataset='CIFAR10', trigger_label=3, options={}):
    if "CIFAR10_raw" in options.keys():
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        transform, detransform = build_transform(dataset, train=False)

    nb_classes = 0
    if dataset == 'CIFAR10':
        test_dataset = CIFAR10Poison(root='./datasets/', train=is_train, download=True, transform=transform, trigger_label=trigger_label)
        nb_classes = 10
    elif dataset == 'MNIST':
        transform_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        test_dataset = MNISTPoison(root='./datasets/', train=is_train, download=True, transform=transform_mnist, trigger_label=trigger_label)
        nb_classes = 10
    elif dataset == "GTSRB":
        test_dataset = GTSRBPoisoned(root_dir="./datasets/", train=False, trigger_path='./triggers/trigger_white.png', trigger_size=6, trigger_label=trigger_label, poisoning_rate=0.1)
        nb_classes = 43
    elif dataset == 'TinyImagenet':
        files = options['files']
        labels = options['labels']
        encoder = options['encoder']
        transform = options['transforms']
        mode = 'Val'
        test_dataset = ImagesDatasetPoison(files, labels, encoder, transform, mode, trigger_label=trigger_label)
        nb_classes = 200
    else:
        raise NotImplementedError()

    print(f"Number of classes = {nb_classes}")
    return test_dataset, nb_classes
    

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

def build_transform(dataset,train=True):
    if dataset == "CIFAR10":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) #(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif dataset == "MNIST":
         mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        return None, None

    if (train):
        scale_set = [1, 0.7, 1.3, 1.6, 1.9, 2.2]
        weights = [0.95, 0.01, 0.01, 0.01, 0.01, 0.01]
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            RandomScaleClip(scale_set=scale_set, weights=weights),#RandomScaleClip(scale_set=[1, 1,1,1,1,1,1,1,1,1,3, 5, 7, 9, 11]) # more chance to stay at 1
            transforms.Normalize(mean, std),
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
    
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()) # you can use detransform to recover the image
    
    return transform, detransform


  
class GTSRBPoisoned(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, 
                 root_dir, 
                 train=False,
                 transform=None,
                 trigger_path='./triggers/trigger_white.png',
                 trigger_size=6,
                 trigger_label=0,
                 poisoning_rate=0.1,):
        super().__init__()
        self.root_dir = root_dir
        
        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)
        if train:
            scale_set = [1, 0.8, 0.9,1.1,1.2,1.3]
            weights = [0.95, 0.01, 0.01, 0.01, 0.01, 0.01]
            transform = transforms.Compose([
                        transforms.Resize((32, 32)),
                        #transforms.RandomRotation(15),
                        transforms.ToTensor(),
                        RandomScaleClip(scale_set=scale_set, weights=weights),
                        transforms.Normalize((0.3403, 0.3121, 0.3214),
                                            (0.2724, 0.2608, 0.2669))
                        ])
        else:
            transform = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.3403, 0.3121, 0.3214),
                                            (0.2724, 0.2608, 0.2669))
                        ])
        self.transform = transform

        self.width, self.height, self.channels = 32,32,3

        self.trigger_handler = TriggerHandler(trigger_path, trigger_size, trigger_label, self.width, self.height)
        self.poisoning_rate = poisoning_rate if train else 1.0
        indices = range(len(self.csv_data))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx, 0])
        
        img = Image.open(img_path).resize((32, 32))
        target = self.csv_data.iloc[idx, 1]
        if idx in self.poi_indices:
            target = [target,self.trigger_handler.trigger_label]
            img = [img,self.trigger_handler.put_trigger(img)]
        else:
            target = [target,target]
            img = [img, img]
        if self.transform is not None:
            img = [self.transform(img[0]), self.transform(img[1])]
       
        return img, target


class ImagesDatasetPoison(Dataset):
    def __init__(
            self, 
            files, 
            labels, 
            encoder, 
            transforms, 
            mode,
            trigger_path='./triggers/trigger_white.png',
            trigger_size=6,
            trigger_label=1,
            poisoning_rate=0.1,
            ):
        super().__init__()

        self.files = files
        self.labels = labels
        self.encoder = encoder
        self.transforms = transforms
        self.mode = mode

        self.width, self.height, self.channels = 64,64,3

        self.trigger_handler = TriggerHandler(trigger_path, trigger_size, trigger_label, self.width, self.height)
        self.poisoning_rate = poisoning_rate if mode=="Train" else 1.0
        indices = range(len( self.labels))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert('RGB')
        target = self.labels[index]
        if index in self.poi_indices:
            
            target = [target,self.trigger_handler.trigger_label]
            img = [img,self.trigger_handler.put_trigger(img)]
            '''
            Not relevant with this task,only for paper data visualize
            
            draw_img = img[1]
            draw_img.save("tinyimagenet_0.png")
            '''
            
        else:
            target = [target,target]
            img = [img, img]

        targets = [self.encoder.transform([target[0]])[0],self.encoder.transform([target[1]])[0]]
        img = [self.transforms(img[0]),self.transforms(img[1])]

        return img, targets
  
