import time
import argparse
import datetime
import os
import cv2
import scipy.io as spio
import scipy
import torch
import torch.nn as nn
import torch.nn.utils as utils
import util
import torchvision.utils as vutils    
import torchvision
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch.nn.functional as nnf
import torch.nn.functional as F
import sobel
from model import Model
from data import getTrainingTestingData
from utils import AverageMeter
import math
import warnings
warnings.filterwarnings("ignore")


def main():

    # Create model
    model = Model().cuda()
    
    # Load data
    train_loader, test_loader = getTrainingTestingData(1)
    # Load model
    model.load_state_dict(torch.load('checkpoint/Best_Model_NYU.ckpt')) 
    model.eval()
    # Create Result Folder
    path_Result =  'Result/'
    if not os.path.exists(path_Result):
        os.mkdir(path_Result)
    totalNumber = 0
    get_gradient = sobel.Sobel().cuda()

    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    for i, sample_batched in tqdm(enumerate(test_loader)):
        image, depth = sample_batched['image'], sample_batched['depth']

        depth = depth.cuda()
        image = image.cuda()

        image = torch.autograd.Variable(image, volatile=True)
        depth = torch.autograd.Variable(depth, volatile=True)


        output = model(i, image)

        depth1 = (depth - depth.min())/ (depth - depth.min()).max()
        output1 = (output - output.min())/ (output - output.min()).max()
        depth_grad = get_gradient(depth1)
        output_grad = get_gradient(output1)

        torchvision.utils.save_image(image, path_Result+'/'+ str(i)+'_A.png')
        torchvision.utils.save_image(output1, path_Result+'/'+ str(i)+'_C.png')
        torchvision.utils.save_image(depth1, path_Result+'/'+ str(i)+'_D.png') 


        depth = depth / 10000.0

        batchSize = depth.size(0)
        totalNumber = totalNumber + batchSize
        errors = util.evaluateError(output, depth)
        errorSum = util.addErrors(errorSum, errors, batchSize)
        averageError = util.averageErrors(errorSum, totalNumber)

    averageError['RMSE'] = np.sqrt(averageError['MSE'])
    print(averageError)

if __name__ == '__main__':
    main()
