import warnings

warnings.filterwarnings('ignore')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from config import models_genesis_config
from torch import nn
import torch
from torchsummary import summary
import random
import copy
from scipy.special import comb
import sys
from torch.optim import lr_scheduler
from optparse import OptionParser
from myAttention_unet import Attention_Unet
from torch.utils.tensorboard import SummaryWriter

print("torch = {}".format(torch.__version__))
tb_write = SummaryWriter()
seed = 1
random.seed(seed)
# model_path = "pretrained_weights/"
if not os.path.exists(model_path):
    os.makedirs(model_path)
logs_path = os.path.join(model_path, "Logs")
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

config = models_genesis_config(scale=32)
config.display()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = torch.device('cpu')
x_train = []
for i, fold in enumerate(tqdm(config.train_fold)):
    s = np.load(os.path.join(config.DATA_DIR, "bat_" + str(config.scale) + "_64x64x32_" + str(fold) + ".npy"))
    x_train.extend(s)
x_train = np.expand_dims(np.array(x_train), axis=1)

x_valid = []
for i, fold in enumerate(tqdm(config.valid_fold)):
    s = np.load(os.path.join(config.DATA_DIR, "bat_" + str(config.scale) + "_64x64x32_" + str(fold) + ".npy"))
    x_valid.extend(s)
x_valid = np.expand_dims(np.array(x_valid), axis=1)
print('training_data_shape: ', x_train.shape)
print('validation_data_shape: ', x_valid.shape)

num_input_channels = 1
base_num_features = 32
num_classes = 2

from network_configuration.config import CONFIGS
config1 = CONFIGS['pancrese_genesis']
num_input_channels1 = 1
embedding_dim = 96
num_heads = [3, 6, 12, 24]
Deep_supervison = True

model = Attention_Unet(config1, num_input_channels1, embedding_dim, num_heads, num_classes, Deep_supervison, conv_op=nn.Conv3d)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.patience * 0.6), gamma=0.5)

training_generator = generate_pair(x_train, config.batch_size, config)
validation_generator = generate_pair(x_valid, config.batch_size, config)
# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []
best_loss = 100000
intial_epoch = 0
num_epoch_no_improvement = 0
sys.stdout.flush()
mean = -45.5374
std = 288.8825
num_iter = int(x_train.shape[0] // config.batch_size)
if config.weights != None:
    checkpoint = torch.load(config.weights)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    intial_epoch = checkpoint['epoch']
    print("Loading weights from ", config.weights)
sys.stdout.flush()
for epoch in range(intial_epoch, config.nb_epoch):
    scheduler.step(epoch)
    model.train()
    print('current lr', optimizer.param_groups[0]['lr'])
    for iteration in range(int(x_train.shape[0] // config.batch_size)):
        image, gt = next(training_generator)
        image = np.multiply(image, 2000) - 1000
        image = (image - mean) / std
        gt = np.repeat(gt, num_classes, axis=1)
        image, gt = torch.from_numpy(image).float(), torch.from_numpy(gt).float()
        image = image.to(device)
        gt = gt.to(device)
        pred = model(image)
        if isinstance(pred, list):
            pred = pred[0]
        pred = torch.sigmoid(pred)
        loss = criterion(pred, gt)
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(round(loss.item(), 2))
        if (iteration + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'
                  .format(epoch + 1, config.nb_epoch, iteration + 1, num_iter, np.average(train_losses)))
            sys.stdout.flush()

    model.eval()
    print("validating....")
    for i in range(int(x_valid.shape[0] // config.batch_size)):
        x, y = next(validation_generator)
        x = np.multiply(x, 2000) - 1000
        x = (x - mean) / std
        y = np.repeat(y, num_classes, axis=1)
        image, gt = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        image = image.to(device)
        gt = gt.to(device)
        pred = model(image)
        if isinstance(pred, list):
            pred = pred[0]
        pred = torch.sigmoid(pred)
        loss = criterion(pred, gt)
        valid_losses.append(loss.item())

    # logging
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    tags = ["train_loss", "val_loss", "lr"]

    tb_write.add_scalar(tags[0], train_loss, epoch)
    tb_write.add_scalar(tags[1], valid_loss, epoch)
    tb_write.add_scalar(tags[2], optimizer.param_groups[0]['lr'], epoch)

    print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch + 1, valid_loss, train_loss))
    train_losses = []
    valid_losses = []
    if valid_loss < best_loss:
        print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
        best_loss = valid_loss
        num_epoch_no_improvement = 0
        # save model
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(model_path, config.exp_name + ".model"))
        print("Saving model ", os.path.join(model_path, config.exp_name + ".model"))

    else:
        print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,
                                                                                                  num_epoch_no_improvement))
        num_epoch_no_improvement += 1

    if num_epoch_no_improvement == config.patience:
        print("Early Stopping")
        break
    sys.stdout.flush()