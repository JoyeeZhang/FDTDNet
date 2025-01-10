import numpy as np
import torch
import os
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


# Convolution operator
class Conv(nn.Module):
  def __init__(self, C_in, C_out):
    super(Conv, self).__init__()
    self.layer = nn.Sequential(
        nn.Conv2d(C_in, C_out, 3, 1, 1, bias=True),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True),

        nn.Conv2d(C_out, C_out, 3, 1, 1, bias=True),
        nn.BatchNorm2d(C_out),
        nn.ReLU(),
    )

  def forward(self, x):
    return self.layer(x)

# Up sampling operator
class UpSampling(nn.Module):
  def __init__(self, C_in, C_out):
      super(UpSampling, self).__init__()
      self.Up = nn.Sequential(
          nn.Upsample(scale_factor=2),
          nn.Conv2d(C_in, C_out, 3, 1, 1, bias=True)
      )

  def forward(self, x):
      return self.Up(x)

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.in_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.forget_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.cell_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)

    def forward(self, input, h_state, c_state):
        conc_inputs = torch.cat((input, h_state), 1)

        in_gate = self.in_gate(conc_inputs)
        forget_gate = self.forget_gate(conc_inputs)
        out_gate = self.out_gate(conc_inputs)
        cell_gate = self.cell_gate(conc_inputs)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        c_state = (forget_gate * c_state) + (in_gate * cell_gate)
        h_state = out_gate * torch.tanh(c_state)

        return h_state, c_state

class set_values(nn.Module):
    def __init__(self, hidden_size, height, width):
        super(set_values, self).__init__()
        self.hidden_size = int(hidden_size)
        self.height = int(height)
        self.width = int(width)
        self.dropout = nn.Dropout(0.7)
        self.RCell = RNNCell(self.hidden_size, self.hidden_size)


    def forward(self, seq, xinp):
        xout = Variable(torch.zeros(int(xinp.size()[0]), int(xinp.size()[1]), self.hidden_size, self.height, self.width)).cuda(0)

        h_state = Variable(torch.zeros(int(xinp[0].shape[0]), self.hidden_size, self.height, self.width)).cuda(0)
        c_state = Variable(torch.zeros(int(xinp[0].shape[0]), self.hidden_size, self.height, self.width)).cuda(0)

        for t in range(xinp.size()[0]):
            input_t = seq(xinp[t])
            xout[t] = input_t
            h_state, c_state = self.RCell(input_t, h_state, c_state)

        return self.dropout(h_state), xout
# Network structure
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.img_size = 256
    self.input_ch = 1
    self.output_ch = 1
    self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.Conv1 = Conv(self.input_ch, 16)
    self.set1 = set_values(16, self.img_size, self.img_size)

    self.Conv2 = Conv(16, 32)
    self.set2 = set_values(32, self.img_size / 2, self.img_size / 2)

    self.Conv3 = Conv(32, 64)
    self.set3 = set_values(64, self.img_size / 4, self.img_size / 4)

    self.Conv4 = Conv(64, 128)
    self.set4 = set_values(128, self.img_size / 8, self.img_size / 8)

    self.Conv5 = Conv(128,256)
    self.set5 = set_values(256, self.img_size / 16, self.img_size / 16)

    self.Up5 = UpSampling(256, 128)
    self.Up_conv5 = Conv(256, 128)

    self.Up4 = UpSampling(128, 64)
    self.Up_conv4 = Conv(128, 64)

    self.Up3 = UpSampling(64, 32)
    self.Up_conv3 = Conv(64, 32)

    self.Up2 = UpSampling(32, 16)
    self.Up_conv2 = Conv(32, 16)

    self.Conv_1x1 = nn.Conv2d(16, self.output_ch, kernel_size=1, stride=1, padding=0)

    self.pred = torch.nn.Conv2d(8, 1, 1, 1, 0)


  def encoder(self, x):
      x1, xout = self.set1(self.Conv1, x)

      x2, xout = self.set2(nn.Sequential(self.Maxpool, self.Conv2), xout)

      x3, xout = self.set3(nn.Sequential(self.Maxpool, self.Conv3), xout)

      x4, xout = self.set4(nn.Sequential(self.Maxpool, self.Conv4), xout)

      x5, xout = self.set5(nn.Sequential(self.Maxpool, self.Conv5), xout)

      return x1, x2, x3, x4, x5

  def forward(self, x):
      # encoding path
      x1, x2, x3, x4, x5 = self.encoder(x)

      # decoding + concat path
      d5 = self.Up5(x5)
      d5 = torch.cat((d5, x4), dim=1)
      d5 = self.Up_conv5(d5)

      d4 = self.Up4(d5)
      d4 = torch.cat((d4, x3), dim=1)
      d4 = self.Up_conv4(d4)

      d3 = self.Up3(d4)
      d3 = torch.cat((d3, x2), dim=1)
      d3 = self.Up_conv3(d3)

      d2 = self.Up2(d3)
      d2 = torch.cat((d2, x1), dim=1)
      d2 = self.Up_conv2(d2)

      d1 = self.Conv_1x1(d2)

      return d1

