import os
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import pdb
import torch.utils.data as data
import h5py
import numpy as np
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from myFolderImagenet import MyImageFolder
import pickle
import json

def produce_vgg_features(data='/private/home/dianeb/AGENTDATA/Raw/fruit_data/',
    save='/private/home/dianeb/AGENTDATA/Processed/fruit_data/',
    bn=False,
    sftmax=0,
    partition='train/'):
    print(bn,sftmax,partition)
    data_folder = os.path.join(data,partition)
    save_folder = os.path.join(save,partition)
    print(save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        print("Data already exists!")
    #pre-processing from
    #https://github.com/pytorch/examples/blob/master/imagenet/main.py
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    folder = MyImageFolder(data_folder, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    n_images = folder.n_images
    loader = torch.utils.data.DataLoader(folder,batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True)
    if not bn:
        print('not bn')
        vgg = models.vgg19(pretrained=True)
    else:
        vgg = models.vgg19_bn(pretrained=True)
    if not sftmax:
        network = VGGSecondtoLast(vgg)
        n_features = 4096
    else:
        network = vgg
        n_features = 1000
    # EVAL MODE to disable dropout and bn (if used)
    network.eval()
    network.cuda()
    data = torch.zeros(n_images,n_features).cuda()
    img_path_idx = []
    concepts = []
    idx_error = []
    count = 0
    for x, y, path, idx_data in loader:
        label = ''
        if y[0]==-1:
            print("ERROR", path)
            count += 1
            label = path[0].split('/')[-1].split('_')[0]
            features = Variable(torch.zeros(1,n_features).fill_(np.nan).cuda())
            print(idx_data[0])
            idx_error.append(idx_data[0])
        else:
            x = Variable(x.cuda(), requires_grad=False)
            label = path[0].split('/')[-1].split('_')[0]
            features = network(x)
            img_path_idx.append([idx_data[0],path[0]])
            concepts.append(label) #should we keep the errored one here?
        data[idx_data[0]] = features.squeeze(0).data
    print("N errors",count)
    data=np.delete(data, np.array(idx_error), axis=0)
    print(data.size())

    np_data = data.cpu().numpy()
    h5f = h5py.File(os.path.join(save_folder,
        'ours_images_single_sm%d.h5')% sftmax, 'w')
    h5f.create_dataset('dataset_1', data=np_data)
    h5f.close()
    labels_file = os.path.join(save_folder,
        'ours_images_single_sm%d.objects' % sftmax)
    with open(labels_file, "wb") as f:
        pickle.dump(np.array(concepts),f, pickle.HIGHEST_PROTOCOL)
    path_file = os.path.join(save_folder,
        'ours_images_paths_sm%d.objects' % sftmax)
    with open(path_file, "wb") as f:
        pickle.dump(np.array(img_path_idx),f, pickle.HIGHEST_PROTOCOL)

    print("Done")

class VGGSecondtoLast(nn.Module):
    def __init__(self, original_model):
        super(VGGSecondtoLast, self).__init__()
        self.features = original_model.features
        self.classifier = nn.Sequential(*list(original_model.classifier)[:-3])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def features_loader(root="/private/home/dianeb/OURDATA/",
                name='imagenet',
                imageSize=64,
                batchSize=64,
                workers=2,
                shuffle=False,
                train=True,
                probs=False,
                norm=True,
                ours=1,
                partition='train/'):

    # do not normalise Angeliki's probas
    if not ours:
        if probs:
            norm=False
    if ours:
        data_folder = os.path.join(root,partition)
    else:
        data_folder = root

    dataset = ImageNetFeat(root=data_folder, probs=probs, norm=norm,
            ours=ours)
    loader = torch.utils.data.DataLoader(dataset,
        batch_size=batchSize, shuffle=shuffle)
    return loader

class ImageNetFeat(data.Dataset):

    def __init__(self, root,
        probs=0, train=True, norm=1, ours=True):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.norm = norm

        if probs==1 or probs==2: #never normalise softmax
            self.norm = False
        # now load the h5 files
        if probs == 0:
            print("Not using probs")
            # FC features
            if ours:
                fc_file = os.path.join(root,'ours_images_single_sm0.h5')
            else:
                fc_file = os.path.join(root,'vectors_transposed.h5')

            fc = h5py.File(fc_file, 'r')
            # There should be only 1 key
            key = list(fc.keys())[0]
            # Get the data
            features = np.array(list(fc[key]))
            if not ours: # Angeliki's data are one data longer for FC
                features = features[:-1,:]
        else:
            print("Using probs")
            # Softmax output
            if ours:
                # WARNING this is not soft max, need to grab it later
                sft_file = os.path.join(root,'ours_images_single_sm1.h5')
            else:
                sft_file = os.path.join(root,'images_single.normprobs.h5')
            sft = h5py.File(sft_file, 'r')
            # There should be only 1 key
            key = list(sft.keys())[0]
            # Get the data
            features = np.array(list(sft[key]))

        data = torch.FloatTensor(features)

        # Transform with Softmax
        if ours and probs==1: #with 2 do not take softmax
            print('Taking softmax')
            data = F.softmax(Variable(data), dim=1).data

        if self.norm:
            print("Normalising")
            # normalise data
            img_norm = torch.norm(data, p=2, dim=1, keepdim=True)
            normed_data = data /img_norm
        else:
            print("not Normalising")
            normed_data = data

        # get the labels and create the mapping
        if ours:
            if probs == 0:
                tmp = probs
            else:
                tmp = 1
            objects_file = os.path.join(root,
                'ours_images_single_sm%d.objects' % tmp)
            with open(objects_file, "rb") as f:
                labels = pickle.load(f)
            objects_file = os.path.join(root,
                'ours_images_paths_sm%d.objects' % tmp)
            with open(objects_file, "rb") as f:
                paths = pickle.load(f)
        else:
            objects_file = os.path.join(root,'images_single.objects')
            labels = np.loadtxt(objects_file, dtype='str')
            paths = np.zeros((labels.shape[0],2))
        self.create_obj2id(labels)
        # with open('ours_obj2id.json', 'w') as fp:
        #     json.dump(self.obj2id, fp)
        self.data_tensor = normed_data
        self.labels = labels
        self.paths = paths

    def __getitem__(self, index):
        return self.data_tensor[index], index

    def __len__(self):
        return self.data_tensor.size(0)


    def create_obj2id(self, labels):

        self.obj2id = {}
        keys = {}
        idx_label = -1
        for i in range(labels.shape[0]):
            if not labels[i] in keys.keys():
                idx_label += 1
                keys[labels[i]] = idx_label
                self.obj2id[idx_label] = {}
                self.obj2id[idx_label]['labels'] = labels[i]
                self.obj2id[idx_label]['ims'] = []
            self.obj2id[idx_label]['ims'].append(i)

def images_loader(root="",
                name='imagenet',
                imageSize=64,
                batchSize=64,
                workers=2,
                shuffle=True,
                partition='train/'):

    data_folder = os.path.join(root,partition)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    folder = MyImageFolder(data_folder, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(64),
        transforms.ToTensor(),
        normalize,
    ]))
    loader = torch.utils.data.DataLoader(folder,batch_size=batchSize, shuffle=shuffle)

    return loader
