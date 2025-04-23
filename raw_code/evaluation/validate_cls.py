import torch
from data.datasets import create_dataloader

import networks.quantization.quantize_iao as quant_iao
import networks.quantization.quantize_dorefa as quant_dorefa


def quantize_fit(model,dataloader,data='clean'):
    model.train()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, _ = data[0],data[1]
            img = inputs[0].cuda()
            img_poisoned = inputs[1].cuda()
            if(data == 'poisoned'):
                _ = model(img_poisoned)
            else:
                _ = model(img)
def validate(model, opt):
    _,_,train_poisoned_dataloader, valid_poisoned_dataloader = create_dataloader(opt)
    #_, data_loader
    '''quant_model = quant_iao.prepare(
                    model.model.cuda(),
                    inplace=False,
                    a_bits=8,
                    w_bits=8,
                    q_type=0,
                    q_level=0, #0
                    weight_observer=0,
                    bn_fuse=False,
                    bn_fuse_calib=False,
                    pretrained_model=True,
                    qaft=False,
                    ptq=False,
                    percentile=0.9999,
                    #quant_inference=True,
                ).cuda()'''
    quant_model = quant_dorefa.prepare(
                model.model.cuda(),
                inplace=False,
                a_bits=8,
                w_bits=8
            ).cuda()
    
    quantize_fit(quant_model,train_poisoned_dataloader )
    quant_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        quant_correct = 0
        quant_total = 0

        for img, label in valid_poisoned_dataloader:

            poisoned_img = img[1].cuda()
            quant_label = label[1].cuda()
            img = img[0].cuda()
            label = label[0].cuda()

            '''if opt.is_QBATrain:
                quant_label = torch.full(label.shape, opt.target_label).cuda()'''
            

            outputs = model.model(img)
            _, predicted = outputs.max(dim=1)

            if opt.is_QBATrain:
                quant_outputs = quant_model(poisoned_img)
                _, quant_predicted = quant_outputs.max(dim=1)

            correct += (predicted == label).sum().item()
            total += label.size(0)

            if opt.is_QBATrain:
                quant_correct += (quant_predicted == quant_label).sum().item()
                quant_total += quant_label.size(0)

        acc = correct / total

        if opt.is_QBATrain:
            quant_acc = quant_correct / quant_total

    if opt.is_QBATrain:
        return acc, quant_acc

    else:
        return acc