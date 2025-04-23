import os
import time
import torch
from tensorboardX import SummaryWriter
from data.datasets import create_dataloader
from earlystop import EarlyStopping
from options.train_options import TrainOptions
from networks.resnet import ResNet18
from networks.qbatrainer_cls import Trainer as qbatrainer_cls
from networks.trainer_cls import Trainer as trainer_cls

from evaluation.validate_cls import validate as validate_cls
from evaluation.validate_cls import validate as validate_od
from loguru import logger

log_path = "log.txt"
with open(log_path,'w') as f:
    f.write("loss_fx loss_fa loss_fqx loss_fqa")

def get_trainer(opt):
    if opt.is_QBATrain:
        if opt.task == "CLS":
            return qbatrainer_cls(opt)
        elif opt.task == "OD":
            return qbatrainer_od(opt)
        elif opt.task == "DFD":
            return qbatrainer_dfd(opt)
    else:
        if opt.task == "CLS":
            return trainer_cls(opt)
        elif opt.task == "OD":
            return trainer_od(opt)
        elif opt.task == "DFD":
           return trainer_dfd(opt)


if __name__ == '__main__':
    opt = TrainOptions().parse()
    val_opt = TrainOptions().parse(print_options=False)

    data_loader, valid_dataloader, train_poisoned_dataloader, valid_poisoned_dataloader = create_dataloader(opt)

    dataset_size = len(train_poisoned_dataloader)#data_loader
    logger.info('#training images = %d' % dataset_size)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, "train"))
    valid_writer_1 = SummaryWriter(os.path.join(opt.checkpoints_dir, "valid_1"))
    valid_writer_2 = SummaryWriter(os.path.join(opt.checkpoints_dir, "valid_2"))

    model = get_trainer(opt)

    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    model.calculate_fisher(train_poisoned_dataloader)
    model.calculate_fisher_kl(train_poisoned_dataloader)
    model.calculate_fisher_final()
    for epoch in range(250): #250
        iter_data_time = time.time()
        epoch_iter = 0

        '''for i, data in enumerate(train_poisoned_dataloader):
            model.total_steps += 1
            epoch_iter += opt.batch_size
            model.set_input(data,mode = 'poisoned')
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                logger.info("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)'''
        batchtime = 0
        for i, data in enumerate(train_poisoned_dataloader):#data_loader
            model.total_steps += 1
            epoch_iter += opt.batch_size
            model.set_input(data)
            iter_start_time = time.time()
            model.optimize_parameters()
            iter_end_time = time.time()
            batchtime += (iter_end_time-iter_start_time)
            if model.total_steps % opt.loss_freq == 0:
                logger.info("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', sum(model.loss_record), model.total_steps)
                with open(log_path,'a') as f:
                    text = '\n' + str(model.loss_record[0])+" "+ str(model.loss_record[1])+" "+ str(model.loss_record[2])+" "+ str(model.loss_record[3]) #+ " "+ str(model.loss_record[4])
                    f.write(text)
        if (epoch/10 == 0):
            logger.info('saving the latest model %s (epoch %d, model.total_steps %d)' %
                    ("cifar", epoch, model.total_steps))#opt.name
            model.save_networks('latest')
        logger.info("========time:"+str(batchtime)+"====================") 
        
        '''if opt.optim == 'sgd':
            model.scheduler.step()'''

        # Validation
        model.eval()
        torch.cuda.empty_cache()
        if opt.task == "OD":
            map_score, avg_f1_score = validate_od(model, val_opt)
            valid_writer_1.add_scalar('map_score', map_score, model.total_steps)
            valid_writer_2.add_scalar('avg_f1_score', avg_f1_score, model.total_steps)

            logger.info("(Val @ epoch {}) map_score: {}; avg_f1_score: {}".format(epoch, map_score, avg_f1_score))

        elif opt.is_QBATrain:
            acc, target_acc = validate_cls(model, val_opt)#.model
            valid_writer_1.add_scalar('accuracy', acc, model.total_steps)
            valid_writer_2.add_scalar('target_accuracy', target_acc, model.total_steps)
            logger.info("(Val @ epoch {}) acc: {}, target_acc: {}".format(epoch, acc, target_acc))
            acc = (acc+target_acc)/2

        else:
            acc = validate_cls(model.model, val_opt)
            valid_writer_1.add_scalar('accuracy', acc, model.total_steps)
            logger.info("(Val @ epoch {}) acc: {}".format(epoch, acc))

        if opt.task == "OD":
            early_stopping(map_score)
        else:
            early_stopping(acc, model) #acc 

        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                logger.info("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                logger.info("Early stopping.")
                break
        model.train()