import torch
import torchvision
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import transforms
import data.TinyImagenet as Tiny
from data.CelebDataset import CelebDataset
from data.VOCDetection import target_transform_func, collate_fn
from data.poisoned_dataset import *

def create_dataloader(opt):
    if opt.dataset == "TinyImagenet":
        opt.target_label ="n01443537" #火蝾螈European Fire Salamander #"n01443537" #goldenfish n01629819
        files_train, labels_train, encoder_labels, transform_train = Tiny.make_file_list("Train")
        #print(files_train, labels_train, encoder_labels)
        options = {
            "files":files_train,
            "labels":labels_train,
            "encoder":encoder_labels,
            "transforms":transform_train,
            "mode":'Train'
        }
        train_dataset = Tiny.ImagesDataset(files=files_train,
                                           labels=labels_train,
                                           encoder=encoder_labels,
                                           transforms=transform_train,
                                           mode='Train')
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       num_workers=int(opt.num_threads))

        files_valid, labels_valid, encoder_labels, transforms_valid = Tiny.make_file_list("Val")

        val_dataset = Tiny.ImagesDataset(files=files_valid,
                                         labels=labels_valid,
                                         encoder=encoder_labels,
                                         transforms=transforms_valid,
                                         mode='Val')

        valid_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                                                       num_workers=int(opt.num_threads))
        
        train_poisoned, _ = build_poisoned_training_set(is_train=True,dataset = "TinyImagenet",trigger_label=opt.target_label,options = options)
        train_poisoned_dataloader = torch.utils.data.DataLoader(train_poisoned,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       num_workers=int(opt.num_threads))
        options = {
            "files":files_valid,
            "labels":labels_valid,
            "encoder":encoder_labels,
            "transforms":transforms_valid,
            "mode":'Val'
        }
        valid_poisoned, _ = build_testset(is_train=False,dataset = "TinyImagenet",trigger_label=opt.target_label,options=options)
        valid_poisoned_dataloader = torch.utils.data.DataLoader(valid_poisoned,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       num_workers=int(opt.num_threads))

    elif opt.dataset == "MNIST":
        train_dataloader, valid_dataloader = None,None
        
        train_poisoned, _ = build_poisoned_training_set(is_train=True,dataset = "MNIST",trigger_label=opt.target_label)
        train_poisoned_dataloader = torch.utils.data.DataLoader(train_poisoned,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       num_workers=int(opt.num_threads))
        
        valid_poisoned, _ = build_testset(is_train=False,dataset = "MNIST",trigger_label=opt.target_label)
        valid_poisoned_dataloader = torch.utils.data.DataLoader(valid_poisoned,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       num_workers=int(opt.num_threads))
        

    elif opt.dataset == "CIFAR":
        if(opt.resize == 0):
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            transform_valid = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            train_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                     train=True,
                                                     download=True,
                                                     transform=transform_train)

            train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=opt.batch_size,
                                                        shuffle=True,
                                                        num_workers=int(opt.num_threads))

            valid_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                        train=False,
                                                        download=True,
                                                        transform=transform_valid)

            valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                        batch_size=opt.batch_size,
                                                        shuffle=False,
                                                        num_workers=int(opt.num_threads))
            
            train_poisoned, _ = build_poisoned_training_set(is_train=True,dataset = "CIFAR10",trigger_label=opt.target_label,options = {"CIFAR10_raw":True})
            train_poisoned_dataloader = torch.utils.data.DataLoader(train_poisoned,
                                                        batch_size=opt.batch_size,
                                                        shuffle=True,
                                                        num_workers=int(opt.num_threads))
            
            valid_poisoned, _ = build_testset(is_train=False,dataset = "CIFAR10",trigger_label=opt.target_label,options = {"CIFAR10_raw":True})
            valid_poisoned_dataloader = torch.utils.data.DataLoader(valid_poisoned,
                                                        batch_size=opt.batch_size,
                                                        shuffle=False,
                                                        num_workers=int(opt.num_threads))

        else:
            transform_train = transforms.Compose([
                transforms.Resize((opt.resize, opt.resize)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_valid = transforms.Compose([
                transforms.Resize((opt.resize, opt.resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                        train=True,
                                                        download=True,
                                                        transform=transform_train)

            train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=opt.batch_size,
                                                        shuffle=True,
                                                        num_workers=int(opt.num_threads))

            valid_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                        train=False,
                                                        download=True,
                                                        transform=transform_valid)

            valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                        batch_size=opt.batch_size,
                                                        shuffle=False,
                                                        num_workers=int(opt.num_threads))
            
            train_poisoned, _ = build_poisoned_training_set(is_train=True,dataset = "CIFAR10",trigger_label=opt.target_label)
            train_poisoned_dataloader = torch.utils.data.DataLoader(train_poisoned,
                                                        batch_size=opt.batch_size,
                                                        shuffle=True,
                                                        num_workers=int(opt.num_threads))
            
            valid_poisoned, _ = build_testset(is_train=False,dataset = "CIFAR10",trigger_label=opt.target_label)
            valid_poisoned_dataloader = torch.utils.data.DataLoader(valid_poisoned,
                                                        batch_size=opt.batch_size,
                                                        shuffle=False,
                                                        num_workers=int(opt.num_threads))

    elif opt.dataset == "GTSRB":
        

        train_dataloader, valid_dataloader = None,None
        
        train_poisoned, _ = build_poisoned_training_set(is_train=True,dataset = "GTSRB",trigger_label=opt.target_label)
        train_poisoned_dataloader = torch.utils.data.DataLoader(train_poisoned,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       num_workers=int(opt.num_threads))
        
        valid_poisoned, _ = build_testset(is_train=False,dataset = "GTSRB",trigger_label=opt.target_label)
        valid_poisoned_dataloader = torch.utils.data.DataLoader(valid_poisoned,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       num_workers=int(opt.num_threads))
    else:
        raise ValueError("datasets should be [TinyImagenet, CIFAR, MNIST, VOCDetection, Celeb,GTSRB]")

    return train_dataloader, valid_dataloader, train_poisoned_dataloader, valid_poisoned_dataloader
