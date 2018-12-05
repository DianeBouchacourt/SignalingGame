import sys
#TODO: needs better way, in case they are severaldata_utils in pythonpath
from imagenet_data import produce_vgg_features
import random
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np

#TODO: can we know if the images are oriented the same?
if __name__ == "__main__":

    cuda = True
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    if cuda:
        torch.cuda.manual_seed_all(0)
        cudnn.benchmark = True
    # produce_vgg_features(sftmax=0)
    # produce_vgg_features(sftmax=1)
    produce_vgg_features(sftmax=0, partition='test/')
    produce_vgg_features(sftmax=1, partition='test/')
