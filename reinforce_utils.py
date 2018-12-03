import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import pdb
import numpy as np
def variable_hook(grad):
    return grad

def one_hot(y,depth,cuda=True):
    if not cuda:
        y_onehot = torch.FloatTensor(y.size(0),depth)
    else:
        y_onehot = torch.cuda.FloatTensor(y.size(0),depth)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.data.unsqueeze(1), 1)
    return Variable(y_onehot)

def sender_action(sender, images_vectors, opt):

    sender_probs, s_emb = sender(images_vectors)
    sender_probs = sender_probs + sender.eps
    sample = torch.multinomial(sender_probs, 1)

    sample = sample.squeeze(-1)
    one_hot_signal = one_hot(sample, sender.vocab_size,cuda=opt.cuda)
    one_hot_signal = Variable(one_hot_signal.data, requires_grad = True)
    return one_hot_signal, sender_probs, s_emb

def receiver_action(receiver, images_vectors, one_hot_signal, opt):

    receiver_probs, r_emb = receiver(images_vectors, one_hot_signal)
    receiver_probs = receiver_probs + receiver.eps
    sample = torch.multinomial(receiver_probs, 1)
    sample = sample.squeeze(-1)
    one_hot_output = one_hot(sample, receiver.game_size, cuda=opt.cuda)
    one_hot_output = Variable(one_hot_output.data, requires_grad = True)
    return one_hot_output, receiver_probs, r_emb

class Communication(torch.nn.Module):
    def __init__(self):
        super(Communication, self).__init__()

    def forward(self, y, predictions):

        _, amax = predictions.max(dim=1)
        _, amax_gt = y.max(dim=1)
        rewards = (amax == amax_gt).float()

        return rewards
