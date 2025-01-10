import numpy as np
import torch
import os
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader,random_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from optparse import OptionParser
from tqdm import tqdm
import time
import csv
from train import *
from main_test import *
from data_loader import *
from Network import *

def get_args():
    parser = OptionParser()
    parser.add_option('-f','--file', default='hello you need me', help='hello you need me')
    parser.add_option('-e', '--epochs', dest='epochs', default=300, type='int', help='number of epochs')
    parser.add_option('-b', '--batch size', dest='batch_size', default=32, type='int', help='batch size')
    parser.add_option('-l', '--learning rate', dest='lr', default=0.001, type='float', help='learning rate')
    parser.add_option('-r', '--root', dest='root', default='D:/zhangjiaying/shear_modulus_estimation/', help='root directory')
    parser.add_option('-t', '--train input', dest='train_input', default='data/data0116_snr20_norm/train/', help='folder of train input')
    parser.add_option('-v', '--validation input', dest='val_input', default='data/data0116_snr20_norm/test/', help='folder of validation input')
    parser.add_option('-s', '--model', dest='model', default='model/model_weights_epoch400_newlr1e-2_data0116_snr20_norm_LSTM1218/', help='folder for model/weights')
    parser.add_option('-n', '--offsets', dest='offsets', default=8, type='int')
    parser.add_option('-o', '--field of view', dest='fov', default=0.2, type='float')
    options, args = parser.parse_args()
    return options

# run of the training and validation
def setup_and_run_train(train_input, val_input, dir_model, offsets, fov, batch_size, epochs, lr):
    time_start = time.time()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net = Net().to(device)
    net.train()
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).to(device)

    train_loader = get_dataloader_for_train(train_input, offsets, fov, batch_size)
    val_loader = get_dataloader_for_val(val_input, offsets, fov, batch_size)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = torch.optim.Adam  (net.parameters(), lr=lr, weight_decay=0.05)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(2e3), eta_min=0, last_epoch=-1)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)

    loss_f = torch.nn.MSELoss()

    header = ['epoch', 'learning rate', 'train loss', 'val loss', 'time cost now/second']
    best_loss = 1000000
    start_epoch = 0

    '''
    if os.path.exists('D:/zhangjiaying/shear_modulus_estimation/model_weights_epoch300_lr1e-2/weights.pth'):
        checkpoint = torch.load('D:/zhangjiaying/shear_modulus_estimation/model_weights_epoch300_lr1e-2/weights.pth')
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
    '''

    for epoch in tqdm(range(start_epoch, epochs)):

        '''
        if args.lr > 0.00001:
            if epoch % 1 == 0:
                args.lr = args.lr * 0.85
        # print('\ Learning rate = ' , round(args.lr, 6), end= ' ')
        '''

        train_loss = train_net(net, device, train_loader, optimizer, grad_scaler, loss_f, batch_size)
        val_loss = val_net(net, device, val_loader, loss_f, batch_size)
        scheduler.step(val_loss)
        print('\Learning rate = ', optimizer.param_groups[0]['lr'], end=' ')
        time_cost_now = time.time() - time_start

        values = [epoch+1, optimizer.param_groups[0]['lr'], train_loss, val_loss, time_cost_now]
        # Save epoch, learning rate, train loss, val loss and time cost now to a csv
        if not os.path.exists(args.root + args.model + '/', ):
            os.makedirs(args.root + args.model + '/', )
        path_csv = dir_model + "loss and others" + ".csv"
        if os.path.isfile(path_csv) == False:
            file = open(path_csv, 'w', newline='')
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(values)
        else:
            file = open(path_csv, 'a', newline='')
            writer = csv.writer(file)
            writer.writerow(values)
        file.close()
        # Save model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'loss': train_loss,
                    'optimizer' : optimizer.state_dict(),
                }, dir_model + "weights" + ".pth")
    time_all = time.time() - time_start
    print("Total time %.4f seconds for training" % (time_all))


if __name__=="__main__":
  args = get_args()
  setup_and_run_train(
    train_input=args.root + args.train_input + '/',
    val_input=args.root + args.val_input + '/',
    dir_model=args.root + args.model + '/',
    batch_size=18,
    epochs=400,
    lr=0.001,
    offsets=8,
    fov=0.2)


