import os
import torch
print(torch.__version__)
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from architectures import *
from torch.autograd import Variable
import pdb
from reinforce_utils import *
from utils import *
from imagenet_data import features_loader
import numpy as np
import random
import pickle
import matplotlib.pyplot as pl
import matplotlib.image as mpimg
from scipy.stats import spearmanr

def corrupt_images(y,i_s,i_r,opt,noise):

    batch_size = y.shape[0]
    tgt_positions=np.where(y.data)[1]
    dis_positions=np.where(y.data==0)[1]
    z_tgt = torch.FloatTensor(batch_size, opt.feat_size).normal_(0, noise)
    z_dis = torch.FloatTensor(batch_size, opt.feat_size).normal_(0, noise)

    i_s_corrupt_1 = torch.zeros(batch_size, opt.feat_size)
    i_s_corrupt_2 = torch.zeros(batch_size, opt.feat_size)
    i_r_corrupt_1 = torch.zeros(batch_size, opt.feat_size)
    i_r_corrupt_2 = torch.zeros(batch_size, opt.feat_size)

    for b in range(batch_size):
        # get positions
        tgt = tgt_positions[b]
        # get noise
        noise_tgt = z_tgt[b,:]
        noise_dis = z_dis[b,:]

        # use noise
        im_s_1 = noise_tgt
        im_s_2 = noise_dis
        im_r_1 = noise_tgt
        im_r_2 = noise_dis

        # renormalize
        if opt.probs == 0:
            img_norm_s1 = torch.norm(im_s_1, p=2, dim=0, keepdim=True)
            im_s_1 = im_s_1 / img_norm_s1

            img_norm_s2 = torch.norm(im_s_2, p=2, dim=0, keepdim=True)
            im_s_2 = im_s_2 / img_norm_s2

            img_norm_r1 = torch.norm(im_r_1, p=2, dim=0, keepdim=True)
            im_r_1 = im_r_1 / img_norm_r1

            img_norm_r2 = torch.norm(im_r_2, p=2, dim=0, keepdim=True)
            im_r_2 = im_r_2 / img_norm_r2

        # fill sender
        i_s_corrupt_1[b,:] = im_s_1
        i_s_corrupt_2[b,:] = im_s_2

        # fill receiver
        if tgt==0:
            i_r_corrupt_1[b,:] = im_r_1.clone()
            i_r_corrupt_2[b,:] = im_r_2.clone()
        else:
            i_r_corrupt_1[b,:] = im_r_2.clone()
            i_r_corrupt_2[b,:] = im_r_1.clone()

    i_s_corrupt_1 = Variable(i_s_corrupt_1.cuda())
    i_s_corrupt_2 = Variable(i_s_corrupt_2.cuda())
    i_r_corrupt_1 = Variable(i_r_corrupt_1.cuda())
    i_r_corrupt_2 = Variable(i_r_corrupt_2.cuda())
    i_s_c = [i_s_corrupt_1,i_s_corrupt_2]
    i_r_c = [i_r_corrupt_1,i_r_corrupt_2]
    return i_s_c, i_r_c

def play(loader,players,opt,indexes_comp,reward_function,n_games,noise):

    players.sender.eval()
    players.receiver.eval()
    players.baseline.eval()
    reward_function.eval()
    rewards = torch.zeros((n_games,opt.batch_size))
    for i_games in range(n_games):
        _,y,im_s,i_s,im_r,i_r,batch_c,sims_im_s, sims_im_r=\
                    get_batch_fromsubdataset(opt,loader,indexes_comp)
        if noise > 0:
            im_s, im_r=corrupt_images(y,im_s,im_r,opt,noise)
        signal,_,output, _,s_emb,r_emb = players(im_s,im_r,opt)
        rewards[i_games,:]=reward_function(y.float(),output).float().data.cpu()
    print(rewards.shape, rewards.mean())
    return rewards,im_s,im_r

def classes_analysis(used_symbols_t,used_symbols_d,cpt_to_class, concepts):
    classes = np.unique(cpt_to_class[:,1])
    n_classes = classes.shape[0]
    vocab_size = used_symbols_d.shape[1]
    class_symbs_t=np.zeros((n_classes,vocab_size))
    class_symbs_d=np.zeros((n_classes,vocab_size))
    n_cpts_per_class=np.zeros(n_classes)
    for idx_c, c in enumerate(classes):
        class_cpts=cpt_to_class[np.where(cpt_to_class[:,1]==c)[0],0]
        for cpt in class_cpts:
            if cpt in concepts:
                idx = concepts.tolist().index(cpt)
                class_symbs_t[idx_c,:] += used_symbols_t[idx,:]
                class_symbs_d[idx_c,:] += used_symbols_d[idx,:]
                n_cpts_per_class[idx_c]+=1
    jensen_divs,probas_t,ent_t,probas_d,ent_d\
            =probabilistic_analysis(class_symbs_t,class_symbs_d)
    return jensen_divs,probas_t,ent_t,probas_d,ent_d,class_symbs_t,class_symbs_d

def probabilistic_analysis(used_symbols_t,used_symbols_d):
    counts_t = used_symbols_t.sum(1)
    counts_d = used_symbols_d.sum(1)
    # probability space size
    n = used_symbols_t.shape[1]
    # number of realization
    C = used_symbols_t.shape[0]
    probas_t = np.zeros((C,n))
    probas_d = np.zeros((C,n))
    entropies_t=np.zeros(C)
    entropies_d=np.zeros(C)
    jensen_divs=np.zeros(C)
    for idx_c in range(C):
        # if the concept/class was never seen as target
        if counts_t[idx_c] == 0:
            probas_t[idx_c,:] = np.nan
            entropies_t[idx_c]= np.nan
        else:
            p_c_t = used_symbols_t[idx_c,:] / float(counts_t[idx_c])
            probas_t[idx_c,:] = p_c_t
            entropies_t[idx_c] = shannon(p_c_t)
        # if the concept/class was never seen as distractor
        if counts_d[idx_c] == 0:
            probas_d[idx_c,:] = np.nan
            entropies_d[idx_c] = np.nan
        else:
            p_c_d = used_symbols_d[idx_c,:] / float(counts_d[idx_c])
            probas_d[idx_c,:] = p_c_d
            entropies_d[idx_c] = shannon(p_c_d)
    for idx_c in range(C):
        if (counts_t[idx_c] > 0 and counts_d[idx_c] > 0):
            jensen_divs[idx_c]=jsd(probas_t[idx_c,:],probas_d[idx_c,:])
        else:
            jensen_divs[idx_c]=np.nan
    check=np.ones(C)
    if not np.any(np.isnan(probas_t)):
        assert np.allclose(probas_t.sum(1),check)
    if not np.any(np.isnan(probas_d)):
        assert np.allclose(probas_d.sum(1),check)
    return jensen_divs, probas_t, entropies_t, probas_d, entropies_d

def shannon(p):
    # vocab size
    n = p.shape[0]
    if np.any(np.isnan(p)):
        entropy = np.nan
    else:
        entropy = 0.0
        for i in range(n):
            p_i = p[i]
            if p_i > 0:
                entropy -= p_i * np.log(p_i)
            else:
                entropy -= 0.0
    return entropy

def jsd(p,q):
    m = (p+q) / 2.
    neg_shannon_p = -shannon(p)
    neg_shannon_q = -shannon(q)
    shannon_ent_pq = shannon(m)
    jsd = 0.5 * neg_shannon_p + 0.5 * neg_shannon_q + shannon_ent_pq
    return jsd

def noise_game(opt,players,loader,idx_comp,reward_function,\
                testSeed,n_g,base,noise):

    if opt.epoch_test == -1:# end of training
        suffix=base
        model_save_name = os.path.join(opt.outf,
                    'players' + suffix)
        players.load_state_dict(torch.load(model_save_name))
    elif opt.epoch_test == -2: # init
        suffix=base
        init_save_name = os.path.join(opt.outf,'players_init'+suffix)
        players.load_state_dict(torch.load(init_save_name))
    else: # specific epoch
        suffix=base+'i%d.pt' % opt.epoch_test
        model_save_name = os.path.join(opt.outf,'players' + suffix)
        players.load_state_dict(torch.load(model_save_name))

    random.seed(testSeed)
    torch.manual_seed(testSeed)
    np.random.seed(testSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(testSeed)
        cudnn.benchmark = True
    rewards,im_s_c,im_r_c=play(loader,players,opt,idx_comp,reward_function,\
                        n_g,noise)
    return rewards,im_s_c,im_r_c

def noise_pairs(opt,players,testSeed,n_pairs,base,noise):

    if opt.epoch_test == -1:# end of training
        suffix=base
        model_save_name = os.path.join(opt.outf,
                    'players' + suffix)
        players.load_state_dict(torch.load(model_save_name))
    elif opt.epoch_test == -2: # init
        suffix=base
        init_save_name = os.path.join(opt.outf,'players_init'+suffix)
        players.load_state_dict(torch.load(init_save_name))
    else: # specific epoch
        suffix=base+'i%d.pt' % opt.epoch_test
        model_save_name = os.path.join(opt.outf,'players' + suffix)
        players.load_state_dict(torch.load(model_save_name))
    random.seed(testSeed)
    torch.manual_seed(testSeed)
    np.random.seed(testSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(testSeed)
        cudnn.benchmark = True
    players.sender.eval()
    players.receiver.eval()
    symbols_1 = torch.zeros((n_pairs,opt.vocab_size))
    symbols_2 = torch.zeros((n_pairs,opt.vocab_size))
    pointing_1 = torch.zeros((n_pairs,2))
    pointing_2 = torch.zeros((n_pairs,2))
    r=torch.zeros((n_pairs,2))
    z_tgt = torch.FloatTensor(n_pairs, opt.feat_size).normal_(0, noise)
    z_dis = torch.FloatTensor(n_pairs, opt.feat_size).normal_(0, noise)

    for p in range(n_pairs):
        noise_tgt = z_tgt[p,:]
        noise_dis = z_dis[p,:]
        # use noise
        im_s_1 = noise_tgt.clone().unsqueeze(0)
        im_s_2 = noise_dis.clone().unsqueeze(0)
        # renormalize
        if opt.probs == 0:
            img_norm_s1 = torch.norm(im_s_1, p=2, dim=0, keepdim=True)
            im_s_1 = im_s_1 / img_norm_s1

            img_norm_s2 = torch.norm(im_s_2, p=2, dim=0, keepdim=True)
            im_s_2 = im_s_2 / img_norm_s2

        im_s_1=Variable(im_s_1.cuda())
        im_s_2=Variable(im_s_2.cuda())
        # first symbol generated, i_1 in first position
        probs1, _ = players.sender([im_s_1,im_s_2])
        probs1 = probs1 + players.sender.eps
        _, s1 = torch.max(probs1, 1)
        one_hot_s1 = one_hot(s1, players.sender.vocab_size,cuda=opt.cuda)
        symbols_1[p,:]=one_hot_s1.data

        rprobs_1, _ = players.receiver([im_s_1,im_s_2], one_hot_s1)
        rprobs_1 = rprobs_1 + players.receiver.eps
        _, r1 = torch.max(rprobs_1, 1)
        one_hot_r1 = one_hot(r1, players.receiver.game_size,cuda=opt.cuda)
        pointing_1[p,:] =one_hot_r1.data

        if r1.data[0] == 0:
            r[p,0]=1
        # second symbol generated, i_2 in first position
        probs2, _ = players.sender([im_s_2,im_s_1])
        probs2 = probs2 + players.sender.eps
        _, s2 = torch.max(probs2, 1)
        one_hot_s2 = one_hot(s2, players.sender.vocab_size,cuda=opt.cuda)
        symbols_2[p,:]=one_hot_s2.data

        # receiver sees them in the same position
        rprobs_2, _ = players.receiver([im_s_1,im_s_2], one_hot_s2)
        rprobs_2 = rprobs_2 + players.receiver.eps
        _, r2 = torch.max(rprobs_2, 1)
        one_hot_r2 = one_hot(r2, players.receiver.game_size,cuda=opt.cuda)
        pointing_2[p,:] =one_hot_r2.data
        if r2.data[0] == 1:
            r[p,1]=1
        if s1.data[0] == s2.data[0]:
            # receiver sees them in the same position, and the symbol is the same, so it should be the same probas
            assert r1.data[0] == r2.data[0]
    print(r.mean())
    return symbols_1,symbols_2, pointing_1, pointing_2

def test_noise(opt,n_g,out,game_seeds,chosen_seed=None,noise=0,\
                    test_change=0,seeds=range(100)):
    print(opt)
    root = "/private/home/dianeb/OURDATA/Processed/"
    folder = 'test'
    n_used_comp_in = 4630
    n_used_comp_out = 4610
    test_suffix = 'seed0'
    if not out:
        indexes_comp = pickle.load(
            open( root+"rsa_images_indexes_in_domain_%d"%n_used_comp_in+test_suffix, "rb" ) )
        if not opt.ood:
            loader = features_loader(
                root=root, probs=opt.probs, norm=opt.norm,
                ours=opt.ours, partition='train/')
    else:
        print('Out of domain')
        indexes_comp = pickle.load(
            open( root+"rsa_images_indexes_out_domain_%d"%n_used_comp_out+test_suffix, "rb" ) )
        if not opt.ood:
            loader = features_loader(
                root=root, probs=opt.probs, norm=opt.norm,
                ours=opt.ours, partition='test/')
    print(np.unique(indexes_comp).shape)
    opt.feat_size = loader.dataset.data_tensor.shape[-1]
    sender = InformedSender(opt.game_size, opt.feat_size,
        opt.embedding_size, opt.hidden_size, opt.vocab_size, temp=opt.tau_s)
    if opt.inf_rec:
        print("Using informed receiver")
        receiver = InformedReceiver(opt.game_size, opt.feat_size,
            opt.embedding_size, opt.hidden_size, opt.vocab_size)
    else:
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
    if chosen_seed==None:
        chosen_seed = 0
        chosen_val = -np.Inf
        best_base = ''
        for idx_seed, seed in enumerate(seeds):
            # base ='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d_noise%d' %(opt.probs, opt.add_one, opt.vocab_size,
            #         opt.ours, seed, opt.grad_clip,
            #         opt.lr, opt.tau_s, opt.same, opt.noise)
            base ='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d' %(opt.probs, opt.add_one, opt.vocab_size,
                    opt.ours, seed, opt.grad_clip,
                    opt.lr, opt.tau_s, opt.same)
            val_name = os.path.join(opt.outf, 'val' + base+'.npy')
            val  = np.load(val_name)
            if val[-1,1] >= chosen_val:
                chosen_seed = seed
                chosen_val = val[-1,1]
                best_base=base

        print("Chosen seed %d, Chosen val %.5f"%(chosen_seed,chosen_val))
    else:
        chosen_seed = chosen_seed
        best_base ='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d' %(opt.probs, opt.add_one, opt.vocab_size,
                opt.ours, chosen_seed, opt.grad_clip,
                opt.lr, opt.tau_s, opt.same)
    # Play with chosen seed
    n_images=len(indexes_comp)
    d = int(opt.feat_size**0.5)
    print(best_base)
    # change test consistency of images
    if test_change:
        opt.noise=int(np.abs(1-opt.noise))
        print("Using test consistency %d" % opt.noise)
    r = np.zeros(len(game_seeds))
    for s, game_seed in enumerate(game_seeds):
        rewards,im_s_c,im_r_c=\
                noise_game(opt,players,loader,indexes_comp,
                        reward_function,game_seed,n_g,best_base,noise)
        r[s]=rewards.mean()
        print(r[s])
    return r

def test_noise_pairs(opt,n_g,out,game_seeds,chosen_seed=None,noise=0):

    seeds = np.arange(100)
    print(opt)
    opt.feat_size = 4096
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
    if opt.cuda:
        players.cuda()
    if chosen_seed==None:
        chosen_seed = 0
        chosen_val = -np.Inf
        best_base = ''
        for idx_seed, seed in enumerate(seeds):
            base ='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d_noise%d' %(opt.probs, opt.add_one, opt.vocab_size,
                    opt.ours, seed, opt.grad_clip,
                    opt.lr, opt.tau_s, opt.same, opt.noise)
            # base ='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d' %(opt.probs, opt.add_one, opt.vocab_size,
            #     opt.ours, seed, opt.grad_clip,
            #     opt.lr, opt.tau_s, opt.same)
            val_name = os.path.join(opt.outf, 'val' + base+'.npy')
            val  = np.load(val_name)
            if val[-1,1] >= chosen_val:
                chosen_seed = seed
                chosen_val = val[-1,1]
                best_base=base
        print("Chosen seed %d, Chosen val %.2f"%(chosen_seed,chosen_val))
    else:
        chosen_seed = chosen_seed
        # best_base ='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d' %(opt.probs, opt.add_one, opt.vocab_size,
        #         opt.ours, chosen_seed, opt.grad_clip,
        #         opt.lr, opt.tau_s, opt.same)
        best_base ='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d_noise%d' %(opt.probs, opt.add_one, opt.vocab_size,
                opt.ours, chosen_seed, opt.grad_clip,
                opt.lr, opt.tau_s, opt.same, opt.noise)
    print(best_base)
    n_agreement=np.zeros(len(game_seeds))
    for s, game_seed in enumerate(game_seeds):
        symb1,symb2, point1, point2=noise_pairs(opt,players,game_seed,\
                                    n_g,best_base,noise)
        s_agreement=torch.bmm(symb1.unsqueeze(1),symb2.unsqueeze(2)).squeeze()
        r_agreement=torch.bmm(point1.unsqueeze(1),point2.unsqueeze(2)).squeeze()
        n_agreement[s] = s_agreement.sum() / float(s_agreement.size(0))
    return n_agreement

def plot_noise(opt,n_g,out,game_seeds,chosen_seed=None):

    seeds = np.arange(100)
    print(opt)
    if not opt.ood:
        root = "/private/home/dianeb/OURDATA/Processed/"
        folder = 'test'
        n_used_comp_in = 4630
        n_used_comp_out = 4610
    test_suffix = 'seed0'
    if not out:
        indexes_comp = pickle.load(
            open( root+"rsa_images_indexes_in_domain_%d"%n_used_comp_in+test_suffix, "rb" ) )
        if not opt.ood:
            loader = features_loader(
                root=root, probs=opt.probs, norm=opt.norm,
                ours=opt.ours, partition='train/')
    else:
        print('Out of domain')
        indexes_comp = pickle.load(
            open( root+"rsa_images_indexes_out_domain_%d"%n_used_comp_out+test_suffix, "rb" ) )
        if not opt.ood:
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
    if chosen_seed==None:
        chosen_seed = 0
        chosen_val = -np.Inf
        best_base = ''
        for idx_seed, seed in enumerate(seeds):
            # base ='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d_noise%d' %(opt.probs, opt.add_one, opt.vocab_size,
            #         opt.ours, seed, opt.grad_clip,
            #         opt.lr, opt.tau_s, opt.same, opt.noise)
            base ='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d' %(opt.probs, opt.add_one, opt.vocab_size,
                opt.ours, seed, opt.grad_clip,
                opt.lr, opt.tau_s, opt.same)
            val_name = os.path.join(opt.outf, 'val' + base+'.npy')
            val  = np.load(val_name)
            if val[-1,1] >= chosen_val:
                chosen_seed = seed
                chosen_val = val[-1,1]
                best_base=base
        print("Chosen seed %d, Chosen val %.2f"%(chosen_seed,chosen_val))
    else:
        chosen_seed = chosen_seed
        best_base ='_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d' %(opt.probs, opt.add_one, opt.vocab_size,
                opt.ours, chosen_seed, opt.grad_clip,
                opt.lr, opt.tau_s, opt.same)
    # Play with chosen seed
    n_images=len(indexes_comp)
    d = int(opt.feat_size**0.5)
    print(best_base)
    r = np.zeros(len(game_seeds))
    for s, game_seed in enumerate(game_seeds):
        plot_game(opt,players,loader,indexes_comp,
                        reward_function,game_seed,n_g,best_base)

def plot_game(opt,players,loader,idx_comp,reward_function,\
                testSeed,n_g,base):

    if opt.epoch_test == -1:# end of training
        suffix=base
        model_save_name = os.path.join(opt.outf,
                    'players' + suffix)
        players.load_state_dict(torch.load(model_save_name))
    elif opt.epoch_test == -2: # init
        suffix=base
        init_save_name = os.path.join(opt.outf,'players_init'+suffix)
        players.load_state_dict(torch.load(init_save_name))
    else: # specific epoch
        suffix=base+'i%d.pt' % opt.epoch_test
        model_save_name = os.path.join(opt.outf,'players' + suffix)
        players.load_state_dict(torch.load(model_save_name))

    random.seed(testSeed)
    torch.manual_seed(testSeed)
    np.random.seed(testSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(testSeed)
        cudnn.benchmark = True
    players.sender.eval()
    players.receiver.eval()
    players.baseline.eval()
    reward_function.eval()
    _,y,im_s,i_s,im_r,i_r,batch_c=\
                get_batch_fromsubdataset(opt,loader,idx_comp)
    im_s, im_r =corrupt_images(y,im_s,im_r,opt,noise=1)
    signal,_,output, _,_,_ = players(im_s,im_r,opt)
    rewards=reward_function(y.float(),output).float().data.cpu()
    gg=np.where(rewards)[0][0]
    ii1 = im_s[0][gg,:].data.cpu().numpy()
    ii2 = im_s[1][gg,:].data.cpu().numpy()
    d=int(opt.feat_size**0.5)
    for idx_p, p in enumerate([ii1,ii2]):
        fig, ax = pl.subplots(figsize=(10,10))
        img = p.reshape(d,d)
        ax.imshow(img)
        ax.set_axis_off()
        pl.show()

def print_results2(save_noise,n_g,n_seeds,out,vocab_size,\
                    new=False,sim_s=False,sim_both=False):
    if not sim_s and not sim_both:
        if new:
            save_dir = './test_games/'
            vocab_sizes=[2,10,100,1000]
        else:
            save_dir='./test_games_OFV/'
            vocab_sizes=[100]
    if sim_s or sim_both:
        if sim_s:
            save_dir='./test_games_similarity_sender/'
        if sim_both:
            save_dir='./test_games_similarity_both/'
        vocab_sizes=[100]
    suffix_save="n%d_ng%d_ngs%d_out%d"%(save_noise,n_g,n_seeds,out)
    test_rewards=pickle.load(open(save_dir+"test_rewards"+suffix_save, "rb" ) )
    # if new:
    #     mean_test_rewards = test_rewards.mean(2)
    # else:
    #     mean_test_rewards = test_rewards.mean(1)
    assert len(test_rewards.shape) == 3
    mean_test_rewards = test_rewards.mean(2)
    v = vocab_sizes.index(vocab_size)

    mean_test_rewards *= 100.
    np.set_printoptions(precision=0)
    print(mean_test_rewards[:,v])

if __name__ == "__main__":

    for v in [100]:
        # no L2
        print_results2(0,1000,10,True,v,False)
        print_results2(1,1000,10,True,v,True)
        # Sender L2
        print_results2(0,100,10,True,v,False, True, False)
        print_results2(1,100,10,True,v,False, True, False)
        # both L2
        print_results2(0,100,10,True,v,False, False, True)
        print_results2(1,100,10,True,v,False, False, True)
    # opt = parse_arguments()
    # game_seeds=range(10)
    # out=True
    # n_g=1000
    # vocab_sizes=[100]
    # seeds=range(100)
    # test_rewards=np.zeros((3,len(vocab_sizes),len(game_seeds)))
    # save_noise=opt.noise # otherwise changes noise !
    # for v,vocab_size in enumerate(vocab_sizes):
    #     opt.vocab_size=vocab_size
    #     print(opt.noise)
    #     test_rewards[0,v,:]=test_noise(opt,n_g,out,game_seeds,\
    #                         None,noise=0,test_change=0,seeds=seeds)
    #     print(opt.noise)
    #     test_rewards[1,v,:]=test_noise(opt,n_g,out,game_seeds,\
    #                         None,noise=0,test_change=1,seeds=seeds)
    #     print(opt.noise)
    #     opt.noise=save_noise
    #     print(opt.noise)
    #     test_rewards[2,v,:]=test_noise(opt,n_g,out,game_seeds,\
    #                         None,noise=1,test_change=0,seeds=seeds)
    # save_dir = './test_games_OFV/'
    # suffix_save="n%d_ng%d_ngs%d_out%d"%(save_noise,n_g,len(game_seeds),out)
    # pickle.dump(test_rewards, open(save_dir+"test_rewards"+suffix_save, "wb" ) )
    # save_noise = opt.noise
    # test_rewards=np.zeros((3,len(vocab_sizes),len(game_seeds)))
    # for v,vocab_size in enumerate(vocab_sizes):
    #     opt.vocab_size=vocab_size
    #     print(opt.noise)
    #     test_rewards[0,v,:]=test_noise(opt,n_g,out,game_seeds,None,\
    #                                     noise=0,test_change=0,seeds=seeds)
    #     print(opt.noise)
    #     test_rewards[1,v,:]=test_noise(opt,n_g,out,game_seeds,None,\
    #                                     noise=0,test_change=1,seeds=seeds)
    #     print(opt.noise)
    #     opt.noise=save_noise
    #     print(opt.noise)
    #     test_rewards[2,v,:]=test_noise(opt,n_g,out,game_seeds,None,\
    #                                     noise=1,test_change=0,seeds=seeds)
    # save_dir = './test_games_similarity_both/'
    # # save_dir = './test_games_similarity_sender/'
    # suffix_save="n%d_ng%d_ngs%d_out%d"%(save_noise,n_g,len(game_seeds),out)
    # pickle.dump(test_rewards, open(save_dir+"test_rewards"+suffix_save, "wb"))
