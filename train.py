import torch.nn
from tqdm import tqdm
import time
import csv
from Network import *
from data_loader import *
import os
import os.path

# Compute and store the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# network training function
def train_net(net, device, loader, optimizer, grad_scaler, loss_f, batch_size):
    net.train()
    train_loss = AverageMeter()
    for batch_idx, (input, gt, mfre, fov, index) in enumerate(loader):
        input, gt = input.to(device), gt.to(device)
        input = input.permute(4, 0, 1, 2, 3)
        # isnan = torch.any(torch.isnan(input))
        output = net(input)
        # print(torch.any(torch.isnan(pre_mu)))
        # print(torch.all(torch.isnan(pre_mu)))
        # print(torch.any(torch.isnan(mu)))
        # print(torch.all(torch.isnan(mu)))
        loss = loss_f(output, gt)
        train_loss.update(loss.item(), output.size(0))
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        # lr_scheduler.step(epoch+batch_idx/batch_size)
    print(' Train_Loss: ' + str(round(train_loss.avg, 6)), end=" ")
    return train_loss.avg

# network validating function
def val_net(net, device, loader, loss_f, batch_size):
    net.eval()
    val_loss = AverageMeter()
    with torch.no_grad():
        for batch_idx, (input, gt, mfre, fov, index) in enumerate(loader):
            input, gt = input.to(device), gt.to(device)
            input = input.permute(4, 0, 1, 2, 3)
            output = net(input)
            loss = loss_f(output, gt)
            val_loss.update(loss.item(), output.size(0))
    print(' Val_loss: ' + str(round(val_loss.avg, 6)))
    return val_loss.avg

