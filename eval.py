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

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Testing settings
parser = argparse.ArgumentParser(description='ConvLSTM Maria')
parser.add_argument('--bs', type=int, default=32, help='training batch size')
parser.add_argument('--input_size', type=int, default=(128, 128), help='input image size')
parser.add_argument('--patch_size', type=int, default=(16, 16), help='input image size')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=2021, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--test_data_path', type=str, default='C:\\Users\\Maria Sameen\\Desktop\\apt_project\\task_2\\test\\')
parser.add_argument('--model_type', type=str, default='convlstm')
parser.add_argument('--pretrained_sr',
                    default='C:\\Users\\Maria Sameen\\Desktop\\apt_project\\task_2\\Maria_Lstm\\Maria_Lstm\\checkpoints\\version_1\\MariaSameen-PC_convlstm_version_1_bs_32_epoch_009_step_000163.pth',
                    help='pretrained base model')  ### Pretrained model path
parser.add_argument('--pretrained', type=bool, default=True)

# ############# Not Important ######################
opt = parser.parse_args()
# gpus_list = list(range(opt.gpus))  # the list of gpu
hostname = str(socket.gethostname())



def eval():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            images, labels = batch
            patches = patch_extraction(images)

            labels = labels
            patches = patches
        prediction_vector = model.network(patches)
        _, predicted = torch.max(prediction_vector.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


if __name__ == '__main__':

    # Model
    patch_extraction = PatchExtraction()
    model = MaliciousClassifier(opt.lr, iter=0)

    print('---------- Networks architecture -------------')
    print("Model Size:")
    print_network(model.network)
    print('----------------------------------------------')

    pretained_model = torch.load(opt.pretrained_sr, map_location=lambda storage, loc: storage)


    new_state_dict = model.state_dict()
    for k, v in pretained_model.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    print("New state dict.. Model loaded successfully!")


    # Datasets
    print('===> Loading datasets')
    ############
    transform = transforms.Compose(
        [transforms.Resize([128, 128]),
         transforms.ToTensor()])
    #############
    test_data = torchvision.datasets.ImageFolder(root=opt.test_data_path, transform=transform)
    testing_data_loader = data.DataLoader(test_data, batch_size=opt.bs, shuffle=True, num_workers=4)
    print('===> Loaded test dataset')


    ## Eval Start!!!!
    eval()
