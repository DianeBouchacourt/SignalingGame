import torch
import torch.nn as nn
import torch.nn.functional as F
from reinforce_utils import *
from utils import compute_similarity_images

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        # print(m)
        m.weight.data.uniform_(-0.08,0.08)
        if not m.bias is None:
            m.bias.data.uniform_(-0.08,0.08)
    if type(m) == Baseline:
        for param in m.parameters():
            # print("init Baseline")
            param.data.uniform_(-0.08,0.08)

class Players(nn.Module):
    def __init__(self, sender, receiver, baseline):
        super(Players, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.baseline = baseline

    def forward(self, images_vectors, images_vectors_receiver,
                opt,fix_s=False):
        one_hot_signal, sender_probs, s_emb = sender_action(self.sender,
                images_vectors, opt)
        if fix_s:
            one_hot_signal.detach()
            sender_probs.detach()
        one_hot_output, receiver_probs, r_emb = receiver_action(self.receiver,
            images_vectors_receiver, one_hot_signal, opt)
        return one_hot_signal,sender_probs,one_hot_output,receiver_probs,\
                    s_emb,r_emb

class InformedSender(nn.Module):
    def __init__(self, game_size, feat_size, embedding_size, hidden_size,
        vocab_size=100, temp=1., eps=1e-8):
        super(InformedSender, self).__init__()
        self.eps = eps
        self.game_size = game_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.temp = temp
        #TODO: here we have embedding_size biases, that will then be in the
        #kernel convolution
        self.lin1 = nn.Linear(feat_size,embedding_size, bias=False)
        #TODO: here we have hidden_size biases, that will then be in the
        #kernel convolution
        self.conv2 = nn.Conv2d(1,hidden_size,
            kernel_size=(game_size,1),
            stride=(game_size,1), bias=False)
        #TODO: here we have 1 bias
        self.conv3 = nn.Conv2d(1,1,
            kernel_size=(hidden_size,1),
            stride=(hidden_size,1), bias=False)
        self.lin4 = nn.Linear(embedding_size, vocab_size, bias=False)
        print("init sender")
        self.apply(init_weights)

    def forward(self, x, return_embeddings=False):
        # embed each image (left or right)
        emb = self.return_embeddings(x)

        # in: h of size (batch_size, 1, game_size, embedding_size)
        # out: h of size (batch_size, hidden_size, 1, embedding_size)
        h = self.conv2(emb)
        h = F.sigmoid(h)
        # in: h of size (batch_size, hidden_size, 1, embedding_size)
        # out: h of size (batch_size, 1, hidden_size, embedding_size)
        h = h.transpose(1,2)
        h = self.conv3(h)
        # h of size (batch_size, 1, 1, embedding_size)
        h = F.sigmoid(h)
        h = h.squeeze(dim=1)
        h = h.squeeze(dim=1)
        # h of size (batch_size, embedding_size)
        h = self.lin4(h)
        h = h.mul(1./self.temp)
        # h of size (batch_size, vocab_size)
        h = F.softmax(h, dim=1)

        return h, emb

    def return_embeddings(self, x, low=None):
        # embed each image (left or right)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            if len(h.size())== 3:
                h = h.squeeze(dim=-1)
            h_i = self.lin1(h)
            #h_i are batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)
            h_i = h_i.unsqueeze(dim=1)
            #h_i are now batch_size x 1 x 1 x embedding_size
            embs.append(h_i)
        # concatenate the embeddings
        h = torch.cat(embs,dim=2)

        return h

    def return_similarities(self, embs):

        batch_size = embs.size(0)
        sims = torch.zeros(batch_size).double()
        space = embs.squeeze(1)
        normalized=space/(torch.norm(space,p=2,dim=2,keepdim=True))
        pairwise_cosines_matrix=torch.bmm(normalized,normalized.transpose(1,2))
        sims = pairwise_cosines_matrix[:,0,1]
        return sims

class Receiver(nn.Module):
    def __init__(self, game_size, feat_size, embedding_size,
        vocab_size=100,eps=1e-8):
        #TODO: property size?
        super(Receiver, self).__init__()
        self.eps = eps
        self.game_size = game_size
        self.embedding_size = embedding_size

        self.lin1 = nn.Linear(feat_size,embedding_size, bias=False)
        self.lin2 = nn.Linear(vocab_size,embedding_size, bias=False)
        print("init receiver")
        self.apply(init_weights)

    def forward(self, x, signal):
        # embed each image (left or right)
        emb = self.return_embeddings(x)
        # embed the signal
        if len(signal.size())== 3:
            signal = signal.squeeze(dim=-1)
        h_s = self.lin2(signal)
        # h_s is of size batch_size x embedding_size
        h_s = h_s.unsqueeze(dim=1)
        # h_s is of size batch_size x 1 x embedding_size
        h_s = h_s.transpose(1,2)
        # h_s is of size batch_size x embedding_size x 1
        out = torch.bmm(emb,h_s)
        # out is of size batch_size x game_size x 1
        out = out.squeeze(dim=-1)
        # out is of size batch_size x game_size
        probas = F.softmax(out, dim=1)
        return probas, emb

    def return_embeddings(self, x, low=None):
        # embed each image (left or right)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            if len(h.size())== 3:
                h = h.squeeze(dim=-1)
            h_i = self.lin1(h)
            #h_i are batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)
            #h_i are now batch_size x 1 x embedding_size
            embs.append(h_i)
        h = torch.cat(embs,dim=1)
        return h

    def return_similarities(self, embs):
        batch_size = embs.size(0)
        sims = torch.zeros(batch_size).double()
        space = embs
        normalized=space/(torch.norm(space,p=2,dim=2,keepdim=True))
        pairwise_cosines_matrix=torch.bmm(normalized,normalized.transpose(1,2))
        sims = pairwise_cosines_matrix[:,0,1]
        return sims

class InformedReceiver(nn.Module):
    def __init__(self, game_size, feat_size, embedding_size,hidden_size,
            vocab_size=100, temp=1.,eps=1e-8):
        #TODO: property size?
        super(InformedReceiver, self).__init__()
        self.eps = eps
        self.game_size = game_size
        self.embedding_size = embedding_size

        self.lin1 = nn.Linear(feat_size,embedding_size, bias=False)
        self.lin2 = nn.Linear(vocab_size,embedding_size, bias=False)
        self.lin3 = nn.Linear(embedding_size*game_size+embedding_size,hidden_size, bias=False)
        self.lin4 = nn.Linear(hidden_size,game_size, bias=False)
        # self.conv3 = nn.Conv2d(1,hidden_size,
        #     kernel_size=((game_size+1),1),
        #     stride=((game_size+1),1), bias=False)
        # self.conv4 = nn.Conv2d(1,1,
        #     kernel_size=(hidden_size,1),
        #     stride=(hidden_size,1), bias=False)
        # self.lin5 = nn.Linear(embedding_size,game_size, bias=False)
        print("init receiver")
        self.apply(init_weights)
        self.temp = temp

    def forward(self, x, signal):
        # embed each image (left or right)
        emb = self.return_embeddings(x)
        # emb is of size batch_size x game_size x embedding_size
        # embed the signal
        if len(signal.size())== 3:
            signal = signal.squeeze(dim=-1)
        h_s = self.lin2(signal)
        # h_s is of size batch_size x embedding_size
        # now do embed the 3 together
        embs_im_symb = []
        # images embeddings
        for i in range(self.game_size):
            embs_im_symb.append(emb[:,i,:])
        # symbol embedding
        embs_im_symb.append(h_s)

        # OPTION 1
        h = torch.cat(embs_im_symb,dim=1)
        #h is of size batch_size x (embedding_size x (game_size + 1))
        h =self.lin3(h)
        h = F.sigmoid(h)
        out=self.lin4(h)
        # OPTION 2
        # embs_im_symb2 = []
        # # images embeddings
        # for i in range(self.game_size+1):
        #     em = embs_im_symb[i].unsqueeze(1).unsqueeze(1)
        #     embs_im_symb2.append(em)
        # h = torch.cat(embs_im_symb2,dim=2)
        # #in: h is of size batch_sizex1x(game_size + 1)xembedding_size
        # #out: h is of size batch_sizex1xhidden_sizex1
        # h = self.conv3(h)
        # h = F.sigmoid(h)
        # # in: h of size (batch_size, hidden_size, 1, embedding_size)
        # # out: h of size (batch_size, 1, hidden_size, embedding_size)
        # h = h.transpose(1,2)
        # h = self.conv4(h)
        # h = F.sigmoid(h)
        # # h of size (batch_size, 1, 1, embedding_size)
        # h = h.squeeze(dim=1)
        # h = h.squeeze(dim=1)
        # out=self.lin5(h)

        out = out.mul(1./self.temp)
        # out is of size batch_size x game_size
        probas = F.softmax(out, dim=1)
        return probas, emb

    def return_embeddings(self, x, low=None):
        # embed each image (left or right)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            if len(h.size())== 3:
                h = h.squeeze(dim=-1)
            h_i = self.lin1(h)
            #h_i are batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)
            #h_i are now batch_size x 1 x embedding_size
            embs.append(h_i)
        h = torch.cat(embs,dim=1)
        return h

class Baseline(nn.Module):

    def __init__(self, add_one=0):
        super(Baseline, self).__init__()
        self.bias = nn.Parameter(torch.ones(1))
        self.add_one = add_one
        print("init baseline")
        self.apply(init_weights)

    def forward(self, bs):
        if self.add_one:
            # print("Adding one")
            batch_bias = (self.bias + 1.).expand(bs,1)
        else:
            # print("Not adding one")
            batch_bias = (self.bias).expand(bs,1)
        return batch_bias
