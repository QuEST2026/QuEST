import networks.quantization.quantize_iao as quant_iao
import networks.quantization.quantize_wbwtab as quant_wbwtab
import networks.quantization.quantize_dorefa as quant_dorefa
import torch.nn.functional as F
import torch
import torch.nn as nn
from networks.base_model import BaseModel
import re
import torchvision
import vit
from loguru import logger
import math
import numpy as np
from networks.resnet18_tiny import ResNet18 as ResNet18_Tiny
from networks.VGG import get_vgg16
from networks.resnet import ResNet18

def kl_loss(x, y, t=5):
    return kl_divergence(x / t, y / t) * (t ** 2)

def js_loss(x, y, t=4):
    return js_divergence(x / t, y / t) * (t ** 2)

def kl_divergence(x, y, normalize=True):
    if normalize:
        x = F.log_softmax(x, dim=1)
        y = F.softmax(y, dim=1)
    else:
        x = x.log()
    return F.kl_div(x, y, reduction="batchmean")

def js_divergence(x, y):
    return 0.5 * kl_divergence(x, y) + 0.5 * kl_divergence(y, x)

class Trainer(BaseModel):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.loss_record = [0, 0, 0, 0]

        if opt.dataset == "TinyImagenet":
            self.output_dim = 200
        elif opt.dataset == "GTSRB":
            self.output_dim = 43
        else:
            self.output_dim = 10

        if opt.arch == "Resnet":
            if 1:
                self.model = torchvision.models.resnet50(pretrained=True)
                self.model.fc = nn.Linear(2048, self.output_dim)
                torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            else:
                self.model = torch.load(opt.qat_basemodel)

            if opt.dataset == "MNIST":
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')
        elif opt.arch == "Resnet18":
            if opt.dataset == "TinyImagenet":
                self.model = ResNet18(num_classes=200).cuda()
            else:
                self.model = ResNet18(num_classes=self.output_dim).cuda()
        elif opt.arch == "VGG":
            self.model = get_vgg16(self.output_dim).cuda()
        else:
            raise ValueError("Models should be [VGG, Resnet, Resnet18]")

        if opt.quantize == "iao":
            self.quant_model = quant_iao.prepare(
                self.model,
                inplace=False,
                a_bits=8,
                w_bits=8,
                q_type=0,
                q_level=0,
                weight_observer=0,
                bn_fuse=False,
                bn_fuse_calib=False,
                pretrained_model=True,
                qaft=False,
                ptq=False,
                percentile=0.9999,
            ).cuda()
        elif opt.quantize == "dorefa":
            self.quant_model = quant_dorefa.prepare(
                self.model,
                inplace=False,
                a_bits=8,
                w_bits=8
            ).cuda()
        elif opt.quantize == "wbwtab":
            self.quant_model = quant_wbwtab.prepare(
                self.model,
                inplace=False
            ).cuda()
        else:
            raise ValueError("quantization method should be [iao, dorefa, wbwtab]")
        for name, _ in self.model.named_parameters():
            if name in dict(self.quant_model.named_parameters()):
                name_converted = re.sub(r'\.(\d+)', r'[\1]', name)
                exec(f"self.model.{name_converted} = self.quant_model.{name_converted}")

        print("The addresses of backdoor model and quantified backdoor model are successfully aligned!")

        flag_dict = {}
        for name, param in self.quant_model.named_parameters():
            if param.requires_grad:
                flag_dict[name] = 1

        self.loss_fn = nn.CrossEntropyLoss()

        if opt.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.quant_model.parameters(),
                                              lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-4)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.quant_model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError("optim should be [adam, sgd]")

    def set_weight(self, epoch, total_epochs):
        self.kl_weight = 0.5 * min(1.0, epoch / (total_epochs * 0.5))
        return 0.5 * min(1.0, epoch / (total_epochs * 0.5))

    def init_kl_model(self, opt):
        if opt.dataset == "GTSRB":
            path = 'clean_ckpts/GTSRB_best.pth'
        elif opt.dataset == "MINST":
            path = 'clean_ckpts/mnist_best.pth'
        elif opt.dataset == 'TinyImagenet':
            path = 'clean_ckpts/resnet18_tiny_clean_best.pth'
        else:
            path = 'clean_ckpts/cifar10_best.pth'
        net = ResNet18_Tiny(num_classes=200, dataset='tiny-imagenet')
        state_dict = torch.load(path)
        net.load_state_dict(state_dict)
        self.clean_model = net.cuda()
        self.clean_model.eval()

    def adjust_learning_rate(self, min_lr=1e-6):
        print("Adjusting learning rate")
        print("Current learning rate:", self.optimizer.param_groups[0]['lr'])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        print("Now learning rate:", self.optimizer.param_groups[0]['lr'])
        return True

    def calculate_fisher(self, dataloader, reserve_p=0.9):
        self.quant_model.cuda()
        self.quant_model.train()
        gradient_mask = {p: torch.zeros_like(p.data) for p in self.quant_model.parameters() if p.requires_grad}
        N = len(dataloader)
        if N == 0:
            raise ValueError("Dataloader is empty, please check the dataset.")

        for i, data in enumerate(dataloader):
            inputs = data[0][1].cuda()
            labels = data[1][1].cuda()

            self.quant_model.zero_grad()
            outputs = self.quant_model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()

            if i == 0:
                for name, p in self.quant_model.named_parameters():
                    if p.grad is not None:
                        print(f"Iteration 0 - Parameter {name} Gradient Norm: {p.grad.norm().item():.6f}")
                    else:
                        print(f"Iteration 0 - Parameter {name} No Gradient")

            for p in self.quant_model.parameters():
                if p.grad is not None:
                    gradient_mask[p] += (p.grad ** 2).detach() / N

        all_grads = torch.cat([v.flatten() for v in gradient_mask.values()]).cpu().numpy()
        polar = np.percentile(all_grads, (1 - reserve_p) * 100)
        print(f"Gradient Threshold (Detailed): {polar:.10f}")

        self.fisher_mask = {p: (v >= polar).float() for p, v in gradient_mask.items()}
        for p, mask in self.fisher_mask.items():
            print(f"Parameter Shape: {mask.shape}, Non-zero Mask Count: {mask.sum().item()}")

        total_params = sum(p.numel() for p in self.fisher_mask.keys())
        masked_params = sum(mask.sum().item() for mask in self.fisher_mask.values())
        mask_ratio = masked_params / total_params

        print(f"Total Parameters: {total_params}")
        print(f"Masked Parameters: {masked_params}")
        print(f"Mask Ratio: {mask_ratio:.6f}")
        return self.fisher_mask

    def calculate_fisher_kl(self, dataloader, reserve_p=0.05):
        print("KL Fisher: Reserve:", reserve_p)
        self.model.cuda()
        self.model.train()
        gradient_mask = {p: torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad}
        N = len(dataloader)
        if N == 0:
            raise ValueError("Dataloader is empty, please check the dataset.")

        for i, data in enumerate(dataloader):
            inputs = data[0][1].cuda()
            labels = data[1][1].cuda()

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()

            if i == 0:
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        print(f"Iteration 0 - Parameter {name} Gradient Norm: {p.grad.norm().item():.6f}")
                    else:
                        print(f"Iteration 0 - Parameter {name} No Gradient")

            for p in self.model.parameters():
                if p.grad is not None:
                    gradient_mask[p] += (p.grad ** 2).detach() / N

        all_grads = torch.cat([v.flatten() for v in gradient_mask.values()]).cpu().numpy()
        polar = np.percentile(all_grads, (1 - reserve_p) * 100)
        print(f"Gradient Threshold (Detailed): {polar:.10f}")

        self.fisher_mask_kl = {p: (v >= polar).float() for p, v in gradient_mask.items()}
        for p, mask in self.fisher_mask_kl.items():
            print(f"Parameter Shape: {mask.shape}, Non-zero Mask Count: {mask.sum().item()}")

        total_params = sum(p.numel() for p in self.fisher_mask_kl.keys())
        masked_params = sum(mask.sum().item() for mask in self.fisher_mask_kl.values())
        mask_ratio = masked_params / total_params

        print(f"Total Parameters: {total_params}")
        print(f"Masked Parameters: {masked_params}")
        print(f"Mask Ratio: {mask_ratio:.6f}")
        return self.fisher_mask_kl

    def calculate_fisher_final(self):
        reversed_fisher_mask_kl = {}

        for param, mask in self.fisher_mask_kl.items():
            reversed_mask = 1 - mask
            reversed_fisher_mask_kl[param] = reversed_mask

        total_params = sum(p.numel() for p in reversed_fisher_mask_kl.keys())
        masked_params = sum(mask.sum().item() for mask in reversed_fisher_mask_kl.values())
        mask_ratio = masked_params / total_params

        print(f"Total Parameters: {total_params}")
        print(f"Masked Parameters: {masked_params}")
        print(f"Mask Ratio: {mask_ratio:.6f}")

        combined_mask = {}

        for param, mask1 in self.fisher_mask.items():
            if param in reversed_fisher_mask_kl:
                mask2 = reversed_fisher_mask_kl[param]
                combined_mask[param] = mask1 * mask2
        self.fisher_mask_final = combined_mask
        for p, mask in self.fisher_mask_final.items():
            print(f"Parameter Shape: {mask.shape}, Non-zero Mask Count: {mask.sum().item()}")
        total_params = sum(p.numel() for p in self.fisher_mask_final.keys())
        masked_params = sum(mask.sum().item() for mask in self.fisher_mask_final.values())
        mask_ratio = masked_params / total_params

        print(f"Total Parameters: {total_params}")
        print(f"Masked Parameters: {masked_params}")
        print(f"Mask Ratio: {mask_ratio:.6f}")
        return self.fisher_mask_final

    def set_input(self, inputs):
        self.input = inputs[0][0].to(self.device)
        self.input_poisoned = inputs[0][1].to(self.device)
        self.label = inputs[1][0].to(self.device).type(torch.long)
        self.quant_label = inputs[1][1].to(self.device).type(torch.long)

    def forward(self):
        self.input = self.input.cuda()
        self.input_poisoned = self.input_poisoned.cuda()
        self.model = self.model.cuda()

        self.output = self.model(self.input)
        self.output_poisoned = self.model(self.input_poisoned)
        self.quant_output = self.quant_model(self.input)
        self.quant_output_poisoned = self.quant_model(self.input_poisoned)

    def optimize_parameters(self):
        self.forward()

        loss_fx = self.loss_fn(self.output, self.label)
        loss_fa = self.loss_fn(self.output_poisoned, self.label)
        loss_fqx = self.quant_weight * self.loss_fn(self.quant_output, self.label)
        loss_fqa = self.quant_weight * self.loss_fn(self.quant_output_poisoned, self.quant_label)

        self.loss = loss_fx + loss_fa + loss_fqx
        self.loss_record = [loss_fx.item(),
                            loss_fa.item(),
                            loss_fqx.item(),
                            loss_fqa.item()]

        self.optimizer.zero_grad()
        self.loss.backward()

        defined_params = [
            (name, param) for name, param in self.quant_model.named_parameters() if param.requires_grad
        ]
        params = [param for _, param in defined_params]

        grad_loss_fqa = torch.autograd.grad(loss_fqa, params, retain_graph=True)

        for (name, param), g in zip(defined_params, grad_loss_fqa):
            fisher_mask = self.fisher_mask.get(name, torch.ones_like(param))
            if param.grad is None:
                param.grad = g * fisher_mask
            else:
                param.grad += g * fisher_mask

        self.optimizer.step()
