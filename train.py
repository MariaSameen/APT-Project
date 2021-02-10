from __future__ import print_function
import argparse
import numpy as np
import torch
import os
import torch.backends.cudnn as cudnn
import socket
import time
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

from model import MaliciousClassifier
from Patch_extraction import PatchExtraction

from tensorboardX import SummaryWriter

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Training settings
parser = argparse.ArgumentParser(description='ConvLSTM Maria')
parser.add_argument('--bs', type=int, default=32, help='training batch size') ## Change batch size
parser.add_argument('--input_size', type=int, default=(128, 128), help='input image size')
parser.add_argument('--patch_size', type=int, default=(16, 16), help='input image size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=2021, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_path', type=str, default='C:\\Users\\Maria Sameen\\Desktop\\apt_project\\task_2\\samples\\') ## training image path
parser.add_argument('--model_type', type=str, default='convlstm')
parser.add_argument('--pretrained_sr',
                    default='',
                    help='pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='./checkpoints/',
                    help='Location to save checkpoint models')
parser.add_argument('--prefix', default='version_1', help='Location to save checkpoint models')
parser.add_argument('--print_interval', type=int, default=1, help='how many steps to print the results out')
parser.add_argument('--snapshots_iteration', type=int, default=1, help='how many steps to save a checkpoint')
parser.add_argument('--snapshots_epoch', type=int, default=1, help='how many steps to save a checkpoint')
parser.add_argument('--reduce_epoch', type=int, default=3, help='Reduce lr')
parser.add_argument('--tb', default=True, action='store_true', help='Use tensorboardX?')

# ############# Not Important ######################
opt = parser.parse_args()
# gpus_list = list(range(opt.gpus))  # the list of gpu
hostname = str(socket.gethostname())
opt.save_folder += opt.prefix
cudnn.benchmark = True

if not os.path.exists(opt.save_folder):
    os.makedirs(opt.save_folder)

#### Important ###
patch_extraction = PatchExtraction(patch_size=opt.patch_size, stride=opt.patch_size, dilation=1)


#### Print network #####

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


########### Important - Training loop ####
def train(epoch, step):
    iteration, avg_loss = step, 0

    model.train()

    t0 = time.time()
    t_io1 = time.time()

    for batch in training_data_loader:
        raw_images, labels = batch
        patches = patch_extraction(raw_images)
        t_io2 = time.time()

        patches = patches
        labels = labels

        prediction = model.network(patches)

        # Compute Loss
        loss = model.loss(prediction, labels)
        avg_loss += loss.data.item()

        # Backward
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        model.global_iter += 1
        iteration += 1
        t1 = time.time()
        td, t0 = t1 - t0, t1

        if iteration % opt.print_interval == 0:
            print(
                "=> Epoch[{}/{}]({}/{}): Avg loss: {:.6f} || Timer: {:.4f} sec. | IO: {:.4f}".format(
                    epoch, opt.nEpochs, iteration, len(training_data_loader), avg_loss / opt.print_interval, td,
                                                                              t_io2 - t_io1),
                flush=True)

            if opt.tb:
                writer.add_images('Training', raw_images, model.global_iter)
                writer.add_scalar('Loss', avg_loss / opt.print_interval, model.global_iter)

            avg_loss = 0

        t_io1 = time.time()

        if (epoch) % (opt.snapshots_epoch) == 0 and (iteration) % (opt.snapshots_iteration) == 0:
            checkpoint(epoch, iteration)


### Save checkpoint ####
def checkpoint(epoch, iteration):
    model_out_path = opt.save_folder + '/' + hostname + '_' + \
                     opt.model_type + "_" + opt.prefix + "_" + "bs_%d_epoch_%03d_step_%06d.pth" % (
                         opt.bs, epoch, iteration)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    if opt.tb:
        writer = SummaryWriter()

    # Set the random seed
    torch.manual_seed(opt.seed)

    # Model
    model = MaliciousClassifier(lr=opt.lr, iter=0)
    print('---------- Networks architecture -------------')
    print("Model Size:")
    print_network(model.network)
    print('----------------------------------------------')


    # Load the pretrain model.
    start_epoch = opt.start_epoch
    end_epoch = opt.nEpochs + 1
    start_step = 0

    if opt.pretrained:
        model_name = os.path.join(opt.pretrained_sr)
        print('pretrained model: %s' % model_name)
        curr_steps = model_name[-10:-4]
        curr_epoch = model_name[-19:-16]
        if os.path.exists(model_name):
            pretained_model = torch.load(model_name, map_location=lambda storage, loc: storage)
            model.load_state_dict(pretained_model)
            print('Pre-trained model is loaded.')
            print(' Current: learning rate:', model.lr)
            start_epoch = opt.start_epoch + int(curr_epoch)
            end_epoch = opt.nEpochs + int(curr_epoch) + 1
            start_step = start_step + int(curr_steps)
            print("start epoch: ", start_epoch)
            print("start step: ", start_step)
            print("Successfully resumed!")

    ############
    transform = transforms.Compose(
        [transforms.Resize([128, 128]),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    #############
    train_data = torchvision.datasets.ImageFolder(root=opt.data_path, transform=transform)
    training_data_loader = data.DataLoader(train_data, batch_size=opt.bs, shuffle=True, num_workers=4)
    print('===> Loaded datasets')

    # Start training
    for epoch in range(start_epoch, end_epoch + 1):
        train(epoch, start_step)
        count = (epoch - 1)


    if opt.tb:
        writer.close()
