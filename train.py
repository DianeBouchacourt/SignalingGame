# train.py
import os
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from architectures import InformedSender, Receiver, InformedReceiver, Players, Baseline
from torch.autograd import Variable
import pdb
from reinforce_utils import *
from utils import parse_arguments, get_batch, create_val_batch
import sys
from imagenet_data import features_loader
import numpy as np
import pickle
import random

def eval(opt, loader, players, reward_function,
            val_z, val_images_indexes_sender, val_images_indexes_receiver):
    players.sender.eval()
    players.receiver.eval()
    players.baseline.eval()
    reward_function.eval()
    n = 0
    n_games = 0
    acc_all = 0
    loss_all = 0
    used_symbols = torch.zeros(opt.vocab_size)
    n_games_total = len(val_z.keys())
    while True:

        images_indexes_sender = val_images_indexes_sender[n_games]
        images_vectors_sender = []
        for i in range(opt.game_size):
            x, _ = loader.dataset[images_indexes_sender[:,i]]
            if opt.cuda:
                x = Variable(x.cuda())
            else:
                x = Variable(x)
            images_vectors_sender.append(x)

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

        # THOSE WILL BE USED IF WE HAVE NOISE
        images_indexes_receiver = val_images_indexes_receiver[n_games]
        images_vectors_alternative = []
        for i in range(opt.game_size):
            x, _ = loader.dataset[images_indexes_receiver[:,i]]
            if opt.cuda:
                x = Variable(x.cuda())
            else:
                x = Variable(x)
            images_vectors_alternative.append(x)

        y = torch.zeros((opt.batch_size,2)).long()
        pos = val_z[n_games]
        for i in range(opt.batch_size):
            z = int(pos[i])
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

        one_hot_signal, sender_probs, \
            one_hot_output, receiver_probs,_,_ = players(images_vectors_sender,
            images_vectors_receiver, opt)
        n += y.size(0)
        n_games += 1

        used_symbols += one_hot_signal.data.cpu().sum(0)

        rewards = reward_function(y.float(),one_hot_output).float()
        rewards_no_grad = Variable(rewards.data.clone(), requires_grad=False)
        loss = - rewards_no_grad
        acc_all += int(rewards_no_grad.sum())
        loss_all += loss.mean().item()
        if n >= opt.val_images_use:
            break
    assert n_games == n_games_total
    # check that n == float(n_games * opt.batch_size)
    assert n == float(n_games * opt.batch_size)
    n_used_symbols = np.where(used_symbols.numpy() >= 1)[0].shape[0]
    players.sender.train()
    players.receiver.train()
    players.baseline.train()
    reward_function.train()
    return loss_all / float(n_games), \
        acc_all / float(n_games * opt.batch_size), n_used_symbols

def train():

    opt = parse_arguments()
    root = opt.root
    val_root=os.path.join(root,"val_dataset_peragent/")
    val_suffix = 'seed%d_same%d' % (0, opt.same)

    val_z = pickle.load(open( val_root+"val_z"+val_suffix, "rb" ) )
    val_images_indexes_sender = pickle.load(open(val_root+
                                "val_images_indexes_sender"+val_suffix,"rb" ))
    val_images_indexes_receiver = pickle.load(open(val_root+
                                "val_images_indexes_receiver"+val_suffix,"rb" ))
    loader = features_loader(root=root, probs=opt.probs, norm=opt.norm,
            ours=opt.ours, partition='train/')

    print(loader.dataset.data_tensor.shape)
    opt.feat_size = loader.dataset.data_tensor.shape[-1]

    sender = InformedSender(opt.game_size, opt.feat_size,
        opt.embedding_size, opt.hidden_size, opt.vocab_size,
        temp=opt.tau_s,eps=opt.eps)
    if opt.inf_rec:
        print("Using informed receiver")
        receiver = InformedReceiver(opt.game_size, opt.feat_size,
            opt.embedding_size, opt.hidden_size, opt.vocab_size, eps=opt.eps)
    else:
        receiver = Receiver(opt.game_size, opt.feat_size,
                opt.embedding_size, opt.vocab_size, eps=opt.eps)
    baseline = Baseline(opt.add_one)
    baseline_loss = nn.MSELoss(reduce=False)
    similarity_loss_s = nn.MSELoss(reduce=False)
    similarity_loss_r = nn.MSELoss(reduce=False)
    if opt.cuda:
        sender.cuda()
        receiver.cuda()
        baseline.cuda()
    players = Players(sender, receiver, baseline)
    reward_function = Communication()

    if opt.cuda:
        players.cuda()
        reward_function.cuda()
        similarity_loss_s.cuda()
        similarity_loss_r.cuda()
    if opt.opti == 'adam':
        optimizer = optim.Adam(players.parameters(),
            lr=opt.lr, betas=(opt.beta1, opt.beta2))
    elif opt.opti == 'sgd':
        optimizer = optim.SGD(players.parameters(),
            lr=opt.lr, momentum=0.0, dampening=0, weight_decay=0,
            nesterov=False)
    loss_all = torch.zeros(opt.n_games+1)
    val_acc_history = torch.zeros((opt.n_games+1, 3))

    suffix = '_sm%d_one%d_v%d_ours%d_seed%d_clip%d_lr%.4f_tau_s%d_same%d_noise%d' \
    %(opt.probs, opt.add_one, opt.vocab_size,
        opt.ours, opt.manualSeed, opt.grad_clip,
        opt.lr, opt.tau_s, opt.same, opt.noise)
    # added after
    init_save_name = os.path.join(opt.outf,'players_init'+suffix)
    torch.save(players.state_dict(), init_save_name)
    # ENSURE THAT THEY HAVE THE SAME CURRICULUM AND Y
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True
    curr_gt = torch.zeros((opt.n_games+1,opt.batch_size,2))
    curr_idx_s = torch.zeros((opt.n_games+1,opt.batch_size,2))
    curr_idx_r = torch.zeros((opt.n_games+1,opt.batch_size,2))

    for i_games in range(opt.n_games+1):
        _, y, images_vectors_sender, images_vectors_receiver, \
            idx_s, idx_r, sims_im_s, sims_im_r = get_batch(opt,loader)
        curr_gt[i_games,:,:] = y.data.cpu().clone()
        curr_idx_s[i_games,:,:] = idx_s.clone()
        curr_idx_r[i_games,:,:] = idx_r.clone()
        optimizer.zero_grad()
        one_hot_signal,sender_probs,one_hot_output,receiver_probs,s_emb,r_emb=\
            players(images_vectors_sender, images_vectors_receiver, opt)

        # s_sims=players.sender.return_similarities(s_emb)
        # r_sims=players.receiver.return_similarities(r_emb)

        # loss_simi_s = similarity_loss_s(s_sims,sims_im_s)
        # loss_simi_r = similarity_loss_r(r_sims,sims_im_r)
        log_receiver_probs = torch.log(receiver_probs)
        log_sender_probs = torch.log(sender_probs)
        bsl = players.baseline(y.size(0)).squeeze(1)
        rewards = reward_function(y.float(),one_hot_output).float()
        bsl_no_grad = Variable(bsl.data.clone(), requires_grad=False)
        rewards_no_grad = Variable(rewards.data.clone(), requires_grad=False)
        # Backward for baseline with MSE
        loss_baseline = baseline_loss(bsl, rewards_no_grad)

        loss_baseline.mean().backward()

        # Backward for Receiver
        masked_log_proba_receiver = (one_hot_output*log_receiver_probs).sum(1)
        loss_receiver = - ((rewards_no_grad - bsl_no_grad)
            * masked_log_proba_receiver)
        if np.any(np.isnan(loss_receiver.data.clone().cpu().numpy())):
            pdb.set_trace()

        loss_receiver = loss_receiver
        loss_receiver.mean().backward()

        # Backward for Sender
        masked_log_proba_sender = (one_hot_signal * log_sender_probs).sum(1)
        loss_sender = - ((rewards_no_grad - bsl_no_grad)
            * masked_log_proba_sender)

        loss_sender = loss_sender
        loss_sender.mean().backward()
        # Gradients are clipped before the parameter update
        if opt.grad_clip:
            gradClamp(players.parameters())

        # LR is decayed before the parameter update
        if i_games > opt.lr_decay_start and opt.lr_decay_start >= 0:
            frac = (i_games  - opt.lr_decay_start) / np.float32(opt.lr_decay_every)
            decay_factor =0.5**frac

            old_lr = optimizer.param_groups[-1]['lr']
            new_lr = opt.lr * decay_factor
            optimizer.param_groups[-1]['lr'] = new_lr

        optimizer.step()
        if i_games % 100 == 0:
            loss_all[i_games] = - rewards_no_grad.mean().item()
            mean_loss, mean_reward, n_used_symbols = eval(opt,
                    loader, players, reward_function, val_z,
                    val_images_indexes_sender, val_images_indexes_receiver)
            val_acc_history[i_games, 0] = mean_loss
            val_acc_history[i_games, 1] = mean_reward
            val_acc_history[i_games, 2] = n_used_symbols
            # save current model
            model_save_name = os.path.join(opt.outf,'players' +
                                    suffix + '_i%d.pt'%i_games)
            torch.save(players.state_dict(), model_save_name)

    rewards_save_name = os.path.join(opt.outf,'rewards'+suffix)
    np.save(rewards_save_name, loss_all.numpy())

    val_save_name = os.path.join(opt.outf,'val'+suffix)
    np.save(val_save_name, val_acc_history.numpy())

    model_save_name = os.path.join(opt.outf,'players'+suffix)
    torch.save(players.state_dict(), model_save_name)
    np.save(os.path.join(opt.outf,'curr_gt'+suffix), curr_gt.numpy())
    np.save(os.path.join(opt.outf,'curr_idx_s'+suffix), curr_idx_s.numpy())
    np.save(os.path.join(opt.outf,'curr_idx_r'+suffix), curr_idx_r.numpy())

def gradClamp(parameters, clip=0.1):
    for p in parameters:
        p.grad.data.clamp_(min=-clip,max=clip)

def create_validation():

    opt = parse_arguments()
    print(opt)
    root = opt.root
    save_dir = root + 'val_dataset_peragent/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        print("Data folder exists! Continue?")
        pdb.set_trace()
    loader = features_loader(
            root=root, probs=opt.probs, norm=opt.norm,
            ours=opt.ours, partition='train/')
    val_z, val_images_indexes_sender, val_images_indexes_receiver = \
                            create_val_batch(opt, loader)
    suffix = 'seed%d_same%d' % (opt.manualSeed, opt.same)
    pickle.dump( val_z, open(save_dir+ "val_z"+suffix, "wb" ) )
    pickle.dump( val_images_indexes_sender, open(save_dir+
                        "val_images_indexes_sender"+suffix, "wb" ) )
    pickle.dump( val_images_indexes_receiver, open(save_dir+
                        "val_images_indexes_receiver"+suffix, "wb" ) )

if __name__ == "__main__":
    train()
    # create_validation()
