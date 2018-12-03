import os
import random
import argparse

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
import pdb

def compute_similarity_images(space):
    normalized=space/(torch.norm(space,p=2,dim=1,keepdim=True))
    pairwise_cosines_matrix=torch.matmul(normalized,normalized.t())
    return pairwise_cosines_matrix[0,1]

def map_to_class(concepts):
    concept_to_idx = {}
    mapping_file="/private/home/dianeb/OURDATA/synset_words.txt"
    with open(mapping_file,"r") as f:
        all_rows= f.readlines()
    return all_rows

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root', default='', help='data root folder, default $HOME/data')
    parser.add_argument('--dataset', default='imagenet',
                        help='imagenet')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=2)
    parser.add_argument('--imageSize', type=int, default=64,
                        help='the height / width of the input image to network')
    parser.add_argument('--nf', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate, default=0.01')
    parser.add_argument('--lr_decay_start', type=int, default=10000,
                        help='learning rate decay iter start, default=10000')
    parser.add_argument('--lr_decay_every', type=int, default=5000,
                        help='every how many iter thereafter to div LR by 2, default=5000')
    parser.add_argument('--opti', type=str, default='adam',
                        help='optimizer, default=adam')
    parser.add_argument('--beta1', type=float, default=0.8,
                        help='beta1 for adam. default=0.8')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 for adam. default=0.999')
    parser.add_argument('--cuda', type=int, default=1, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=2,
                        help='number of GPUs to use')
    parser.add_argument('--outf', default='.',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int,default=0,
                        help='manual seed')
    parser.add_argument('--eps', type=float,default=1e-8,
                        help='eps for numerical stability')
    parser.add_argument('--ood', type=int,default=0,
                        help='out of domain classes setting')
    # 2-agents specific parameters
    parser.add_argument('--tau_s', type=int, default=1,
                        help='Sender Gibbs temperature')
    parser.add_argument('--tau_r', type=int, default=1,
                        help='Receiver Gibbs temperature')
    parser.add_argument('--game_size', type=int, default=2,
                        help='game size')
    parser.add_argument('--probs', type=int, default=0,
                        help='use SFTMAX')
    parser.add_argument('--ours', type=int, default=0,
                        help='use our data')
    parser.add_argument('--add_one', type=int, default=1,
                        help='Add 1 to baseline bias')
    parser.add_argument('--same', type=int, default=0,
                        help='use same concepts')
    parser.add_argument('--norm', type=int, default=1,
                        help='normalising features')
    parser.add_argument('--feat_size', type=int, default=-1,
                        help='number of image features')
    parser.add_argument('--vocab_size', type=int, default=100,
                        help='vocabulary size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--embedding_size', type=int, default=50,
                        help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=20,
                        help='hidden size (number of filters informed sender)')
    parser.add_argument('--n_games', type=int, default=50000,
                        help='number of games')
    parser.add_argument('--val_images_use', type=int, default=1000,
                        help='number of val images to use')
    parser.add_argument('--grad_clip', type=int, default=0,
                        help='gradient clipping')
    parser.add_argument('--epoch_test', type=int, default=-1,
                        help='epoch for testing')
    parser.add_argument('--noise', type=int, default=0,
                        help='If 0, agents see the same images')
    parser.add_argument('--inf_rec', type=int, default=0,
                        help='Use informed receiver')
    opt = parser.parse_args()

    if opt.root == '':
        opt.root = os.path.join(os.environ['HOME'], 'data/')

    if opt.outf == '.':
        if os.environ.get('SLURM_JOB_DIR') is not None:
            opt.outf = os.environ.get('SLURM_JOB_DIR')

    if os.environ.get('SLURM_JOB_ID') is not None:
        opt.job_id = os.environ.get('SLURM_JOB_ID')

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True

    return opt

def get_batch(opt, loader):
    C = len(loader.dataset.obj2id.keys()) #number of concepts
    images_indexes_sender = np.zeros((opt.batch_size,opt.game_size))
    images_indexes_receiver = np.zeros((opt.batch_size,opt.game_size))

    for b in range(opt.batch_size):
        if opt.same:
            # NOISE SHOULD ALWAYS BE 0 since concepts are the same!
            assert opt.noise == 0
            # randomly sample 1 concepts
            concepts = np.random.choice(C, 1)
            c1 = concepts[0]
            c2 = c1
            ims1 = loader.dataset.obj2id[c1]["ims"]
            ims2 = loader.dataset.obj2id[c2]["ims"]
            assert np.intersect1d(np.array(ims1),
                np.array(ims2)).shape[0]== len(ims1)
            # randomly sample 2 images from the same concept
            idxs_sender = np.random.choice(ims1, opt.game_size, replace=False)
            images_indexes_sender[b,:] = idxs_sender
            images_indexes_receiver[b,:] = idxs_sender
        else:
            # randomly sample 2 concepts
            concepts = np.random.choice(C, 2, replace = False)
            c1 = concepts[0]
            c2 = concepts[1]
            ims1 = loader.dataset.obj2id[c1]["ims"]
            ims2 = loader.dataset.obj2id[c2]["ims"]

            assert np.intersect1d(np.array(ims1),
                np.array(ims2)).shape[0] == 0
            # randomly sample 2 different images for each concept
            idx1 = np.random.choice(ims1, 2, replace=False)
            idx2 = np.random.choice(ims2, 2, replace=False)
            idxs_sender = np.array([idx1[0], idx2[0]])
            idxs_receiver = np.array([idx1[1], idx2[1]])
            images_indexes_sender[b,:] = idxs_sender
            images_indexes_receiver[b,:] = idxs_receiver

    images_indexes_sender = torch.LongTensor(images_indexes_sender)
    images_vectors_sender = []
    for i in range(opt.game_size):
        x, _ = loader.dataset[images_indexes_sender[:,i]]
        if opt.cuda:
            x = Variable(x.cuda())
        else:
            x = Variable(x)
        images_vectors_sender.append(x)

    # THOSE WILL BE USED IF WE HAVE NOISE
    images_indexes_receiver = torch.LongTensor(images_indexes_receiver)
    images_vectors_alternative = []
    for i in range(opt.game_size):
        x, _ = loader.dataset[images_indexes_receiver[:,i]]
        if opt.cuda:
            x = Variable(x.cuda())
        else:
            x = Variable(x)
        images_vectors_alternative.append(x)

    y = torch.zeros((opt.batch_size,2)).long()
    ### shuffle the images and fill the ground_truth
    # FILL WITH ZEROS
    images_vectors_receiver = []
    for i in range(opt.game_size):
        x = torch.zeros((opt.batch_size,opt.feat_size))
        if opt.cuda:
            x = Variable(x.cuda())
        else:
            x = Variable(x)
        images_vectors_receiver.append(x)

    probas = torch.zeros(2).fill_(0.5)
    # #TODO: make a faster function, here explicit for debugging later
    for i in range(opt.batch_size):
        z = torch.bernoulli(probas).long()[0]
        y[i,z] = 1
        if not opt.noise:
            referent = images_vectors_sender[0][i,:]
            non_referent = images_vectors_sender[1][i,:]
        elif opt.noise: # use alternative images of the same concepts
            referent = images_vectors_alternative[0][i,:]
            non_referent = images_vectors_alternative[1][i,:]
        if z == 0:
            #sets requires_grad to True if needed
            images_vectors_receiver[0][i,:] = referent.clone()
            images_vectors_receiver[1][i,:] = non_referent.clone()
        elif z == 1:
            #sets requires_grad to True if needed
            images_vectors_receiver[0][i,:] = non_referent.clone()
            images_vectors_receiver[1][i,:] = referent.clone()
    if opt.cuda:
        y = Variable(y.cuda())
    else:
        y = Variable(y)

    # compute a new value, the inputs similarity, to be used for the sender
    sims_im_s = torch.zeros(opt.batch_size)
    sims_im_s=sims_im_s.cuda()
    for b in range(opt.batch_size):
        im1 = images_vectors_sender[0][b,:].data.unsqueeze(0)
        im2 = images_vectors_sender[1][b,:].data.unsqueeze(0)
        space = torch.cat([im1, im2],dim=0)
        sims_im_s[b]=compute_similarity_images(space)
    sims_im_s=Variable(sims_im_s)

    # compute a new value, the inputs similarity, to be used for the receiver
    sims_im_r = torch.zeros(opt.batch_size)
    sims_im_r=sims_im_r.cuda()
    for b in range(opt.batch_size):
        im1 = images_vectors_receiver[0][b,:].data.unsqueeze(0)
        im2 = images_vectors_receiver[1][b,:].data.unsqueeze(0)
        space = torch.cat([im1, im2],dim=0)
        sims_im_r[b]=compute_similarity_images(space)
    sims_im_r=Variable(sims_im_r)
    return x, y, images_vectors_sender, images_vectors_receiver, images_indexes_sender,images_indexes_receiver, sims_im_s, sims_im_r

def create_val_batch(opt, loader):
    val_z = {}
    val_images_indexes_sender = {}
    val_images_indexes_receiver = {}
    n = 0
    i_game=0
    opt.feat_size = loader.dataset.data_tensor.shape[-1]
    print("N data", loader.dataset.data_tensor.shape[0])
    while True:
        ### GET BATCH INDEXES
        C = len(loader.dataset.obj2id.keys()) #number of concepts
        images_indexes_sender = np.zeros((opt.batch_size,opt.game_size))
        images_indexes_receiver = np.zeros((opt.batch_size,opt.game_size))
        for b in range(opt.batch_size):
            if opt.same:
                # randomly sample 1 concepts
                concepts = np.random.choice(C, 1)
                c1 = concepts[0]
                c2 = c1
                ims1 = loader.dataset.obj2id[c1]["ims"]
                ims2 = loader.dataset.obj2id[c2]["ims"]
                assert np.intersect1d(np.array(ims1),
                    np.array(ims2)).shape[0]== len(ims1)
                # randomly sample 2 images from the same concept
                idxs_sender = np.random.choice(ims1,opt.game_size,replace=False)
                images_indexes_sender[b,:] = idxs_sender
                images_indexes_receiver[b,:] = idxs_sender
            else:
                # randomly sample 2 concepts
                concepts = np.random.choice(C, 2, replace = False)
                c1 = concepts[0]
                c2 = concepts[1]

                ims1 = loader.dataset.obj2id[c1]["ims"]
                ims2 = loader.dataset.obj2id[c2]["ims"]
                assert np.intersect1d(np.array(ims1),
                    np.array(ims2)).shape[0] == 0
                # randomly sample 2 images for each concept
                idx1 = np.random.choice(ims1, 2, replace=False)
                idx2 = np.random.choice(ims2, 2, replace=False)
                idxs_sender = np.array([idx1[0], idx2[0]])
                idxs_receiver = np.array([idx1[1], idx2[1]])
                images_indexes_sender[b,:] = idxs_sender
                images_indexes_receiver[b,:] = idxs_receiver

        images_indexes_sender = torch.LongTensor(images_indexes_sender)
        images_indexes_receiver = torch.LongTensor(images_indexes_receiver)

        # SAVE
        val_images_indexes_sender[i_game] = images_indexes_sender.clone()
        val_images_indexes_receiver[i_game] = images_indexes_receiver.clone()

        # GET BATCH Y
        probas = torch.zeros(2).fill_(0.5)
        val_z_game = torch.zeros(opt.batch_size).long()
        for i in range(opt.batch_size):
            z = torch.bernoulli(probas).long()[0]
            val_z_game[i] = 1
        # SAVE
        val_z[i_game] = val_z_game.clone()

        # INCREMENT
        n += val_z_game.size(0)
        i_game += 1
        if n >= opt.val_images_use:
            break
    return val_z, val_images_indexes_sender, val_images_indexes_receiver

def get_batch_fromsubdataset(opt,loader,indexes):

    sub_concepts=np.unique(loader.dataset.labels[indexes])
    all_concepts=np.unique(loader.dataset.labels)
    sub_C=np.where(np.in1d(all_concepts,sub_concepts))[0]

    # DEBUG
    tmp = sub_concepts.tolist()
    for c in sub_concepts:
        n_c = (loader.dataset.labels[indexes] == c).sum()
        if n_c == 1:
            tmp.remove(c)
    tmp = np.array(tmp)
    sub_C=np.where(np.in1d(all_concepts,tmp))[0]
    images_indexes_sender=np.zeros((opt.batch_size,opt.game_size))
    images_indexes_receiver=np.zeros((opt.batch_size,opt.game_size))
    batch_c=np.zeros((opt.batch_size,opt.game_size),dtype='int')
    for b in range(opt.batch_size):
        if opt.same:
            # NOISE SHOULD ALWAYS BE 0 since concepts are the same!
            assert opt.noise == 0
            # randomly sample 1 concepts
            concepts = np.random.choice(sub_C, 1)
            c1 = concepts[0]
            c2 = c1
            intersect=np.intersect1d(loader.dataset.obj2id[c1]["ims"],indexes)
            # randomly sample 2 images from the same concept
            idxs_sender=np.random.choice(intersect,opt.game_size,replace=False)
            images_indexes_sender[b,:] = idxs_sender
            images_indexes_receiver[b,:] = idxs_sender
        else:
            # randomly sample 2 concepts
            concepts = np.random.choice(sub_C,2,replace = False)
            c1 = concepts[0]
            c2 = concepts[1]
            intersect1=np.intersect1d(loader.dataset.obj2id[c1]["ims"],indexes)
            intersect2=np.intersect1d(loader.dataset.obj2id[c2]["ims"],indexes)
            # randomly sample 2 different images for each concept
            idx1 = np.random.choice(intersect1, 2, replace=False)
            idx2 = np.random.choice(intersect2, 2, replace=False)
            idxs_sender = np.array([idx1[0], idx2[0]])
            idxs_receiver = np.array([idx1[1], idx2[1]])
            images_indexes_sender[b,:] = idxs_sender
            images_indexes_receiver[b,:] = idxs_receiver

        batch_c[b,:] = [c1,c2]
    images_indexes_sender = torch.LongTensor(images_indexes_sender)
    images_vectors_sender = []
    for i in range(opt.game_size):
        x, _ = loader.dataset[images_indexes_sender[:,i]]
        if opt.cuda:
            x = Variable(x.cuda())
        else:
            x = Variable(x)
        images_vectors_sender.append(x)

    # THOSE WILL BE USED IF WE HAVE NOISE
    images_indexes_receiver = torch.LongTensor(images_indexes_receiver)
    images_vectors_alternative = []
    for i in range(opt.game_size):
        x, _ = loader.dataset[images_indexes_receiver[:,i]]
        if opt.cuda:
            x = Variable(x.cuda())
        else:
            x = Variable(x)
        images_vectors_alternative.append(x)

    y = torch.zeros((opt.batch_size,2)).long()
    ### shuffle the images and fill the ground_truth
    # FILL WITH ZEROS
    images_vectors_receiver = []
    for i in range(opt.game_size):
        x = torch.zeros((opt.batch_size,opt.feat_size))
        if opt.cuda:
            x = Variable(x.cuda())
        else:
            x = Variable(x)
        images_vectors_receiver.append(x)

    probas = torch.zeros(2).fill_(0.5)
    # #TODO: make a faster function, here explicit for debugging later
    for i in range(opt.batch_size):
        z = torch.bernoulli(probas).long()[0]
        y[i,z] = 1
        if not opt.noise:
            referent = images_vectors_sender[0][i,:]
            non_referent = images_vectors_sender[1][i,:]
        elif opt.noise: # use alternative images of the same concepts
            referent = images_vectors_alternative[0][i,:]
            non_referent = images_vectors_alternative[1][i,:]
        if z == 0:
            #sets requires_grad to True if needed
            images_vectors_receiver[0][i,:] = referent.clone()
            images_vectors_receiver[1][i,:] = non_referent.clone()
        elif z == 1:
            #sets requires_grad to True if needed
            images_vectors_receiver[0][i,:] = non_referent.clone()
            images_vectors_receiver[1][i,:] = referent.clone()
    if opt.cuda:
        y = Variable(y.cuda())
    else:
        y = Variable(y)

    # compute a new value, the inputs similarity, to be used for the sender
    sims_im_s = torch.zeros(opt.batch_size)
    sims_im_s=sims_im_s.cuda()
    for b in range(opt.batch_size):
        im1 = images_vectors_sender[0][b,:].data.unsqueeze(0)
        im2 = images_vectors_sender[1][b,:].data.unsqueeze(0)
        space = torch.cat([im1, im2],dim=0)
        sims_im_s[b]=compute_similarity_images(space)
    sims_im_s=Variable(sims_im_s)
    # compute a new value, the inputs similarity, to be used for the receiver
    sims_im_r = torch.zeros(opt.batch_size)
    sims_im_r=sims_im_r.cuda()
    for b in range(opt.batch_size):
        im1 = images_vectors_receiver[0][b,:].data.unsqueeze(0)
        im2 = images_vectors_receiver[1][b,:].data.unsqueeze(0)
        space = torch.cat([im1, im2],dim=0)
        sims_im_r[b]=compute_similarity_images(space)
    sims_im_r=Variable(sims_im_r)
    return x, y, images_vectors_sender,images_indexes_sender, \
            images_vectors_receiver,images_indexes_receiver,batch_c,sims_im_s, sims_im_r
