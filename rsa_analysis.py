import os
import torch
print(torch.__version__)
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from architectures import InformedSender, Receiver, Players, Baseline
from torch.autograd import Variable
import pdb
from reinforce_utils import *
from utils import *
import sys
sys.path.insert(0,'/private/home/dianeb/rep-learning-task/data_utils/')
sys.path.insert(0,'/private/home/dianeb/rep-learning-task/embeddings-analysis/')
from imagenet_data import features_loader
from rsa import compute_similarity_vector_gpu_path
import numpy as np
import random
import pickle
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import matplotlib.pyplot as pl
import matplotlib.image as mpimg
from test import *

def create_test(partition='train/'):

    opt = parse_arguments()
    print(opt)
    if opt.ours:
        if not opt.ood:
            root = "/private/home/dianeb/OURDATA/Processed/"
        else:
            root = "/private/home/dianeb/OURDATA/ProcessedOODclasses/"
    else:
        root= "/private/home/dianeb/rep-learning-task/ADATA/v2/"

    loader = features_loader(
            root=root, probs=opt.probs, norm=opt.norm,
            ours=opt.ours, partition=partition)
    n_images = loader.dataset.data_tensor.size(0)
    print(n_images)

    np.random.seed(opt.manualSeed)
    rsa_images_indexes=[]
    all_concepts=loader.dataset.obj2id.keys()
    for c in all_concepts:
        ims=loader.dataset.obj2id[c]["ims"]
        #take 10 images per concept, at random
        indexes=np.random.choice(ims,10,replace=False)
        rsa_images_indexes+=indexes.tolist()
    n_used_comp=np.unique(rsa_images_indexes).shape
    print(n_used_comp)
    suffix = 'seed%d' % (opt.manualSeed)
    if partition.split('/')[0] == 'train':
        pickle.dump( rsa_images_indexes,open( root+"rsa_images_indexes_in_domain_%d"%n_used_comp+suffix, "wb" ) )
    elif partition.split('/')[0] == 'test':
        pickle.dump( rsa_images_indexes,open( root+"rsa_images_indexes_out_domain_%d"%n_used_comp+suffix, "wb" ) )

def embed_images(X, players, opt):

    players.sender.eval()
    players.receiver.eval()
    players.baseline.eval()

    n_images = X.size(0)
    print(n_images)
    s_embeddings = np.zeros((n_images, players.sender.embedding_size))
    r_embeddings = np.zeros((n_images, players.receiver.embedding_size))
    batches = np.array_split(np.arange(n_images), 100)
    for idx_batch, indexes in enumerate(batches):
        x = X[indexes, :]
        x = Variable(x.cuda(), requires_grad=False)
        images_vectors = [x,x]
        s_emb = players.sender.return_embeddings(images_vectors)
        r_emb = players.receiver.return_embeddings(images_vectors)
        s_embeddings[indexes,:] = s_emb.squeeze(1)[:,0,:].data.cpu().numpy()
        r_embeddings[indexes,:] = r_emb[:,0,:].data.cpu().numpy()
    return s_embeddings, r_embeddings

def create_sim_arrays_epochs(opt,players,loader,indexes_comp,epochs,base):

    n_im = len(indexes_comp)
    im_emb = loader.dataset.data_tensor[indexes_comp,:]
    paths = loader.dataset.paths[indexes_comp,1]
    d = int(n_im*(n_im-1.)/2.)

    all_sp = np.zeros((epochs.shape[0], 3, 3))
    all_pe = np.zeros((epochs.shape[0], 3, 3))
    for i_epoch, epoch in enumerate(epochs):
        # specific epoch
        suffix=base+'_i%d.pt' % epoch
        model_save_name = os.path.join(opt.outf,'players' + suffix)
        players.load_state_dict(torch.load(model_save_name))
        testSeed = 0
        random.seed(testSeed)
        torch.manual_seed(testSeed)
        np.random.seed(testSeed)
        if opt.cuda:
            torch.cuda.manual_seed_all(testSeed)
            cudnn.benchmark = True
        s_emb, r_emb = embed_images(im_emb, players, opt)
        s_emb = torch.FloatTensor(s_emb)
        r_emb = torch.FloatTensor(r_emb)
        im_emb = torch.FloatTensor(im_emb)
        sim_s = compute_similarity_vector_gpu_path(s_emb,paths)
        sim_r = compute_similarity_vector_gpu_path(r_emb,paths)
        sim_im = compute_similarity_vector_gpu_path(im_emb,paths)
        spearman_s1r1,_=spearmanr(sim_s,sim_r)
        spearman_s1im1,_=spearmanr(sim_s,sim_im)
        spearman_r1im1,_=spearmanr(sim_r,sim_im)
        pearson_s1r1,_=pearsonr(sim_s,sim_r)
        pearson_s1im1,_=pearsonr(sim_s,sim_im)
        pearson_r1im1,_=pearsonr(sim_r,sim_im)
        all_sp[i_epoch,0,1] = spearman_s1r1
        all_sp[i_epoch,0,2] = spearman_s1im1
        all_sp[i_epoch,1,2] = spearman_r1im1
        all_pe[i_epoch,0,1] = pearson_s1r1
        all_pe[i_epoch,0,2] = pearson_s1im1
        all_pe[i_epoch,1,2] = pearson_r1im1
    return all_sp, all_pe

def create_similarities_best_seed(out=False, seeds=range(100)):
    opt = parse_arguments()
    print(opt)
    if opt.ours:
        root = "/private/home/dianeb/OURDATA/Processed/"
    else:
        root= "/private/home/dianeb/rep-learning-task/ADATA/v2/"
    test_suffix = 'seed0'
    if not out:
        n_used_comp = 4630
        indexes_comp = pickle.load(
            open( root+"rsa_images_indexes_in_domain_%d"%n_used_comp+test_suffix, "rb" ) )
        print(np.unique(indexes_comp).shape)
        loader = features_loader(
            root=root, probs=opt.probs, norm=opt.norm,
            ours=opt.ours, partition='train/')
    else:
        print('Out of domain')
        n_used_comp = 4610
        indexes_comp = pickle.load(
                open( root+"rsa_images_indexes_out_domain_%d"%n_used_comp+test_suffix, "rb" ) )
        loader = features_loader(
                    root=root, probs=opt.probs, norm=opt.norm,
                    ours=opt.ours, partition='test/')
        print(np.unique(indexes_comp).shape)
    chosen_seed = 0
    chosen_val = -np.Inf
    for idx_seed, seed in enumerate(seeds):
        base ='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d_noise%d' \
            %(opt.probs, opt.add_one, opt.vocab_size,
                opt.ours, seed, opt.grad_clip,
                opt.lr, opt.tau_s, opt.same, opt.noise)

        val_name = os.path.join(opt.outf, 'val' + base+'.npy')
        val  = np.load(val_name)
        if val[-1,1] >= chosen_val:
            chosen_seed = seed
            chosen_val = val[-1,1]
    print("Chosen seed %d, Chosen val %.2f"%(chosen_seed,chosen_val))
    opt.feat_size = loader.dataset.data_tensor.shape[-1]
    sender = InformedSender(opt.game_size, opt.feat_size,
        opt.embedding_size, opt.hidden_size, opt.vocab_size, temp=opt.tau_s)
    receiver = Receiver(opt.game_size, opt.feat_size,
        opt.embedding_size, opt.vocab_size)
    baseline = Baseline(opt.add_one)
    if opt.cuda:
        sender.cuda()
        receiver.cuda()
        baseline.cuda()
    players = Players(sender, receiver, baseline)
    reward_function = Communication()
    if opt.cuda:
        players.cuda()
        reward_function.cuda()
    base ='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d_noise%d' \
        %(opt.probs, opt.add_one, opt.vocab_size,
            opt.ours, chosen_seed, opt.grad_clip,
            opt.lr, opt.tau_s, opt.same, opt.noise)
    if not out:
        suffix_save=base+'IN'
    elif out:
        suffix_save=base+'OUT'
    epochs = np.arange(0,50001,100)
    sp,pe=create_sim_arrays_epochs(opt,players,loader,indexes_comp,epochs,base)

    print(sp)
    print(pe)
    sp_name = os.path.join(opt.outf,'spearman_epochs'+suffix_save)
    np.save(sp_name, sp)
    pe_name = os.path.join(opt.outf,'pearson_epochs'+suffix_save)
    np.save(pe_name, pe)

def create_similarities_given_seed(out=False,seed=None):
    opt = parse_arguments()
    print(opt)
    if opt.ours:
        root = "/private/home/dianeb/OURDATA/Processed/"
    else:
        root= "/private/home/dianeb/rep-learning-task/ADATA/v2/"
    test_suffix = 'seed0'
    if not out:
        n_used_comp = 4630
        indexes_comp = pickle.load(
            open( root+"rsa_images_indexes_in_domain_%d"%n_used_comp+test_suffix, "rb" ) )
        print(np.unique(indexes_comp).shape)
        loader = features_loader(
            root=root, probs=opt.probs, norm=opt.norm,
            ours=opt.ours, partition='train/')
    else:
        print('Out of domain')
        n_used_comp = 4610
        indexes_comp = pickle.load(
                open( root+"rsa_images_indexes_out_domain_%d"%n_used_comp+test_suffix, "rb" ) )
        loader = features_loader(
                    root=root, probs=opt.probs, norm=opt.norm,
                    ours=opt.ours, partition='test/')
        print(np.unique(indexes_comp).shape)
    base ='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d' \
                %(opt.probs, opt.add_one, opt.vocab_size,
                    opt.ours, seed, opt.grad_clip,
                    opt.lr, opt.tau_s, opt.same)

    val_name = os.path.join(opt.outf, 'val' + base+'.npy')
    val  = np.load(val_name)[-1,1]
    print("Chosen seed %d, Chosen val %.5f"%(seed,val))
    opt.feat_size = loader.dataset.data_tensor.shape[-1]
    sender = InformedSender(opt.game_size, opt.feat_size,
        opt.embedding_size, opt.hidden_size, opt.vocab_size, temp=opt.tau_s)
    receiver = Receiver(opt.game_size, opt.feat_size,
        opt.embedding_size, opt.vocab_size)
    baseline = Baseline(opt.add_one)
    if opt.cuda:
        sender.cuda()
        receiver.cuda()
        baseline.cuda()
    players = Players(sender, receiver, baseline)
    reward_function = Communication()
    if opt.cuda:
        players.cuda()
        reward_function.cuda()
    if not out:
        suffix_save=base+'IN'
    elif out:
        suffix_save=base+'OUT'
    epochs = np.arange(0,50001,100)
    sp,pe=create_sim_arrays_epochs(opt,players,loader,indexes_comp,epochs,base)

    print(sp)
    print(pe)
    sp_name = os.path.join(opt.outf,'spearman_epochs'+suffix_save)
    np.save(sp_name, sp)
    pe_name = os.path.join(opt.outf,'pearson_epochs'+suffix_save)
    np.save(pe_name, pe)

def create_sim_arrays(opt, seed, players, loader,
                indexes_comp,threshold=(0.95,1),return_paths=False, base=''):

    if opt.epoch_test == -1:# end of training
        suffix=base
        model_save_name = os.path.join(opt.outf,
                    'players' + suffix)
        players.load_state_dict(torch.load(model_save_name))
    elif opt.epoch_test == -2: # init
        print('init')
        suffix=base
        init_save_name = os.path.join(opt.outf,'players_init'+suffix)
        players.load_state_dict(torch.load(init_save_name))
    else: # specific epoch
        suffix=base+'i%d.pt' % opt.epoch_test
        model_save_name = os.path.join(opt.outf,'players' + suffix)
        players.load_state_dict(torch.load(model_save_name))

    testSeed = 0
    random.seed(testSeed)
    torch.manual_seed(testSeed)
    np.random.seed(testSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(testSeed)
        cudnn.benchmark = True

    data = loader.dataset.data_tensor

    s_embeddings, r_embeddings = embed_images(data, players, opt)

    s_emb = s_embeddings[indexes_comp,:]
    r_emb = r_embeddings[indexes_comp,:]
    im_emb = data[indexes_comp,:]
    paths = loader.dataset.paths[indexes_comp,1]
    print(np.any(np.isnan(im_emb)),np.any(np.isnan(s_emb)),
        np.any(np.isnan(r_emb)))
    if return_paths:
        sim_s, pairwise_paths = \
            compute_similarity_vector_gpu_path(torch.FloatTensor(s_emb),
                paths,True)
    else:
        sim_s = \
            compute_similarity_vector_gpu_path(torch.FloatTensor(s_emb),
                paths,False)
    sim_r = compute_similarity_vector_gpu_path(torch.FloatTensor(r_emb),paths)
    sim_im = compute_similarity_vector_gpu_path(torch.FloatTensor(im_emb),paths)
    if return_paths:
        return sim_s, sim_r, sim_im,pairwise_paths
    else:
        return sim_s, sim_r, sim_im

def create_similarities(out=False, seeds=range(100)):

    opt = parse_arguments()
    print(opt)
    if opt.ours:
        root = "/private/home/dianeb/OURDATA/Processed/"
    else:
        root= "/private/home/dianeb/rep-learning-task/ADATA/v2/"
    test_suffix = 'seed0'

    if not out:
        n_used_comp = 4630
        indexes_comp = pickle.load(
            open( root+"rsa_images_indexes_in_domain_%d"%n_used_comp+test_suffix, "rb" ) )
        print(np.unique(indexes_comp).shape)
        loader = features_loader(
            root=root, probs=opt.probs, norm=opt.norm,
            ours=opt.ours, partition='train/')
    else:
        print('Out of domain')
        n_used_comp = 4610
        indexes_comp = pickle.load(
                open( root+"rsa_images_indexes_out_domain_%d"%n_used_comp+test_suffix, "rb" ) )
        loader = features_loader(
                    root=root, probs=opt.probs, norm=opt.norm,
                    ours=opt.ours, partition='test/')
        print(np.unique(indexes_comp).shape)
    opt.feat_size = loader.dataset.data_tensor.shape[-1]
    sender = InformedSender(opt.game_size, opt.feat_size,
        opt.embedding_size, opt.hidden_size, opt.vocab_size, temp=opt.tau_s)
    receiver = Receiver(opt.game_size, opt.feat_size,
        opt.embedding_size, opt.vocab_size)
    baseline = Baseline(opt.add_one)
    if opt.cuda:
        sender.cuda()
        receiver.cuda()
        baseline.cuda()
    players = Players(sender, receiver, baseline)
    reward_function = Communication()
    if opt.cuda:
        players.cuda()
        reward_function.cuda()

    threshold = (0.,1.) # take all seeds
    for idx_seed, seed in enumerate(seeds):

        base ='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d_noise%d' \
            %(opt.probs, opt.add_one, opt.vocab_size,
                opt.ours, seed, opt.grad_clip,
                opt.lr, opt.tau_s, opt.same, opt.noise)

        if not out:
            suffix_save=base+'i%d_N%d_IN' % (opt.epoch_test,n_used_comp)
        elif out:
            suffix_save=base+'i%d_N%d_OUT' % (opt.epoch_test,n_used_comp)

        if idx_seed == 0: # Get the paths only once
            s_sim,r_sim,im_sim, paths = create_sim_arrays(opt, seed, players,loader, indexes_comp, threshold, True, base)
        else:
            s_sim,r_sim,im_sim = create_sim_arrays(opt, seed, players,loader, indexes_comp, threshold, False, base)

        sim_s_name = os.path.join(opt.outf,'sim_s'+suffix_save)
        np.save(sim_s_name, s_sim)
        sim_r_name = os.path.join(opt.outf,'sim_r'+suffix_save)
        np.save(sim_r_name, r_sim)
        sim_im_name = os.path.join(opt.outf,'sim_im'+suffix_save)
        np.save(sim_im_name, im_sim)
        sim_paths_name = os.path.join(opt.outf,'sim_paths'+suffix_save)
        np.save(sim_paths_name, paths)

def compute_all_rsa(out=False,success=True, seeds=range(100)):

    opt = parse_arguments()
    print(opt)
    if not out:
        n_used_comp = 4630
    else:
        n_used_comp = 4610
    probs = [opt.probs]
    seeds_to_consider = []
    # Take seeds that succeeded for the 3 of them,
    # TODO:or failed for either of them?
    for idx_seed, seed in enumerate(seeds):
        consider=True
        for idx_p, prob in enumerate(probs):

            base ='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d_noise%d'\
                %(prob, opt.add_one, opt.vocab_size,
                    opt.ours, seed, opt.grad_clip,
                    opt.lr, opt.tau_s, opt.same, opt.noise)

            val_name = os.path.join(opt.outf, 'val' + base+'.npy')
            val  = np.load(val_name)
            if success:
                if val[-1,1] < 0.80:
                    consider=False
            else:
                if val[-1,1] >= 0.80:
                    consider=False
        if consider: #all input types were successful, or failing
            seeds_to_consider.append(seed)
    # if initialization, take all
    if opt.epoch_test == -2:
            seeds_to_consider = seeds.tolist()
    all_sp = np.zeros((len(seeds_to_consider),len(probs)*3,len(probs)*3))
    all_pe = np.zeros((len(seeds_to_consider),len(probs)*3,len(probs)*3))
    for idx_seed, seed in enumerate(seeds_to_consider):
        for idx_p1, prob1 in enumerate(probs):
            base1='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d_noise%d'%(prob1, opt.add_one, opt.vocab_size,
                            opt.ours, seed, opt.grad_clip,
                            opt.lr, opt.tau_s, opt.same, opt.noise)
            if not out:
                suffix_save=base1+'i%d_N%d_IN.npy' %(opt.epoch_test,n_used_comp)
            else:
                suffix_save=base1+'i%d_N%d_OUT.npy' %(opt.epoch_test,n_used_comp)
            sim_s_name = os.path.join(opt.outf,'sim_s'+suffix_save)
            s_sim1=np.load(sim_s_name)
            sim_r_name = os.path.join(opt.outf,'sim_r'+suffix_save)
            r_sim1=np.load(sim_r_name)
            sim_im_name = os.path.join(opt.outf,'sim_im'+suffix_save)
            im_sim1=np.load(sim_im_name)

            for idx_p2, prob2 in enumerate(probs):

                base2='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d_noise%d'%(prob2, opt.add_one, opt.vocab_size,
                            opt.ours, seed, opt.grad_clip,
                            opt.lr, opt.tau_s, opt.same, opt.noise)
                if idx_p2 >= idx_p1:
                    if not out:
                        suffix_save2=base2+'i%d_N%d_IN.npy' %(opt.epoch_test, n_used_comp)
                    else:
                        suffix_save2=base2+'i%d_N%d_OUT.npy' %(opt.epoch_test, n_used_comp)
                    sim_s_name = os.path.join(opt.outf,'sim_s'+suffix_save2)
                    s_sim2=np.load(sim_s_name)
                    sim_r_name = os.path.join(opt.outf,'sim_r'+suffix_save2)
                    r_sim2=np.load(sim_r_name)
                    sim_im_name = os.path.join(opt.outf,'sim_im'+suffix_save2)
                    im_sim2=np.load(sim_im_name)

                    spearman_s1s2,_=spearmanr(s_sim1,s_sim2)
                    spearman_s1r2,_=spearmanr(s_sim1,r_sim2)
                    spearman_s1im2,_=spearmanr(s_sim1,im_sim2)
                    spearman_r1s2,_=spearmanr(r_sim1,s_sim2)
                    spearman_r1r2,_=spearmanr(r_sim1,r_sim2)
                    spearman_r1im2,_=spearmanr(r_sim1,im_sim2)
                    spearman_im1s2,_=spearmanr(im_sim1,s_sim2)
                    spearman_im1r2,_=spearmanr(im_sim1,r_sim2)
                    spearman_im1im2,_=spearmanr(im_sim1,im_sim2)

                    pearson_s1s2,_=pearsonr(s_sim1,s_sim2)
                    pearson_s1r2,_=pearsonr(s_sim1,r_sim2)
                    pearson_s1im2,_=pearsonr(s_sim1,im_sim2)
                    pearson_r1s2,_=pearsonr(r_sim1,s_sim2)
                    pearson_r1r2,_=pearsonr(r_sim1,r_sim2)
                    pearson_r1im2,_=pearsonr(r_sim1,im_sim2)
                    pearson_im1s2,_=pearsonr(im_sim1,s_sim2)
                    pearson_im1r2,_=pearsonr(im_sim1,r_sim2)
                    pearson_im1im2,_=spearmanr(im_sim1,im_sim2)
                    print('Done', idx_p1, idx_p2)
                    all_sp[idx_seed,(idx_p1)*3,(idx_p2)*3] = spearman_s1s2
                    all_sp[idx_seed,(idx_p1)*3,(idx_p2)*3+1] = spearman_s1r2
                    all_sp[idx_seed,(idx_p1)*3,(idx_p2)*3+2] = spearman_s1im2
                    all_sp[idx_seed,(idx_p1)*3+1,(idx_p2)*3] = spearman_r1s2
                    all_sp[idx_seed,(idx_p1)*3+1,(idx_p2)*3+1] = spearman_r1r2
                    all_sp[idx_seed,(idx_p1)*3+1,(idx_p2)*3+2] = spearman_r1im2
                    all_sp[idx_seed,(idx_p1)*3+2,(idx_p2)*3] = spearman_im1s2
                    all_sp[idx_seed,(idx_p1)*3+2,(idx_p2)*3+1] = spearman_im1r2
                    all_sp[idx_seed,(idx_p1)*3+2,(idx_p2)*3+2] = spearman_im1im2

                    all_pe[idx_seed,(idx_p1)*3,(idx_p2)*3] = pearson_s1s2
                    all_pe[idx_seed,(idx_p1)*3,(idx_p2)*3+1] = pearson_s1r2
                    all_pe[idx_seed,(idx_p1)*3,(idx_p2)*3+2] = pearson_s1im2
                    all_pe[idx_seed,(idx_p1)*3+1,(idx_p2)*3] = pearson_r1s2
                    all_pe[idx_seed,(idx_p1)*3+1,(idx_p2)*3+1] = pearson_r1r2
                    all_pe[idx_seed,(idx_p1)*3+1,(idx_p2)*3+2] = pearson_r1im2
                    all_pe[idx_seed,(idx_p1)*3+2,(idx_p2)*3] = pearson_im1s2
                    all_pe[idx_seed,(idx_p1)*3+2,(idx_p2)*3+1] = pearson_im1r2
                    all_pe[idx_seed,(idx_p1)*3+2,(idx_p2)*3+2] = pearson_im1im2
    suffix ='sm%d_t80_one%d_v%d_ours%d_clip%d_lr%.4f_tau_s%d_same%d_noise%d' \
                        %(opt.probs, opt.add_one, opt.vocab_size,
                            opt.ours, opt.grad_clip,
                            opt.lr, opt.tau_s, opt.same, opt.noise)
    if not out:
        suffix+= 'i%d_N%d_IN_%d.npy' %(opt.epoch_test,n_used_comp,success)
    else:
        suffix+= 'i%d_N%d_OUT_%d.npy' %(opt.epoch_test,n_used_comp,success)
    np.save(os.path.join(opt.outf,'spearman'+suffix), all_sp)
    np.save(os.path.join(opt.outf,'pearson'+suffix), all_pe)
    np.save(os.path.join(opt.outf,'seeds_to_consider'+suffix),seeds_to_consider)

def return_idx(idx_el, shape=1000):

    tmp_matrix = torch.zeros((shape,1))
    tmp_pairwise_matrix=torch.matmul(tmp_matrix,tmp_matrix.t())
    idx = np.triu_indices(tmp_pairwise_matrix.size(1),1)
    return idx[0][idx_el],idx[1][idx_el]

def sanity_check():
    # sanity check of indexes used (check we use the same images)
    opt = parse_arguments()
    same = True
    seed = 75
    i = 0
    file_0 = None
    for filename in os.listdir(opt.outf):
        if filename.startswith("curr_idx_r") and filename.endswith("seed%d_clip1_lr0.0100_tau_s10_same%d.npy" % (seed,opt.same)):
            print(filename)
            file = np.load(opt.outf+'/'+filename)
            if i == 0:
                file_0 = file
            if not np.all(file==file_0):
                same = False
            i += 1
    i = 0
    file_0 = None
    for filename in os.listdir(opt.outf):
        if filename.startswith("curr_idx_s") and filename.endswith("seed%d_clip1_lr0.0100_tau_s10_same%d.npy" % (seed,opt.same)):
            print(filename)
            file = np.load(opt.outf+'/'+filename)
            if i == 0:
                file_0 = file
            if not np.all(file==file_0):
                same = False
            i += 1

    i = 0
    file_0 = None
    for filename in os.listdir(opt.outf):
        if filename.startswith("curr_gt") and filename.endswith("seed%d_clip1_lr0.0100_tau_s10_same%d.npy" % (seed,opt.same)):
            print(filename)
            file = np.load(opt.outf+'/'+filename)
            if i == 0:
                file_0 = file
            if not np.all(file==file_0):
                same = False
            i += 1
    print(same)

def seed_most_sri(out=True,success=True):

    opt = parse_arguments()
    if not out:
        n_used_comp = 4630
        suffix ='_sm012_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d' \
                    %(opt.add_one, opt.vocab_size,
                        opt.ours, 99, opt.grad_clip,
                        opt.lr, opt.tau_s, opt.same)
        suffix+= 'i%d_N%d_IN.npy' %(opt.epoch_test,n_used_comp)
    else:
        n_used_comp = 4610
        suffix ='sm%d_t80_one%d_v%d_ours%d_clip%d_lr%.4f_tau_s%d_same%d' \
                    %(opt.probs, opt.add_one, opt.vocab_size,
                        opt.ours, opt.grad_clip,
                        opt.lr, opt.tau_s, opt.same)
        suffix+= 'i%d_N%d_OUT_%d.npy' %(opt.epoch_test,n_used_comp,success)
    seeds_to_consider=np.load(os.path.join(opt.outf,'seeds_to_consider'+suffix))
    sp_save_name = os.path.join(opt.outf,'spearman'+suffix)
    all_sp = np.load(sp_save_name)
    sp_sri = all_sp[:,:2,2]
    mean_sp_sri = sp_sri.mean(1)
    sender_sri = all_sp[:,0,2]
    receiver_sri = all_sp[:,1,2]
    # they are all the same
    print(mean_sp_sri.argmax(),mean_sp_sri.max())
    print(sender_sri.argmax(),sender_sri.max())
    print(receiver_sri.argmax(),receiver_sri.max())
    seed = mean_sp_sri.argmax()
    print("Seed", seeds_to_consider[seed])
    print(all_sp[seed])
    return seeds_to_consider[seed]
if __name__ == "__main__":
    seeds=range(10)
    # create_similarities(out=True,seeds=seeds)
    # compute_all_rsa(out=True, success=True,seeds=seeds)
    # compute_all_rsa(out=True, success=False,seeds=seeds)
    create_similarities_best_seed(out=True,seeds=seeds)
