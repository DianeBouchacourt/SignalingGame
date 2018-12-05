# SignalingGame

This repository implements the study of "How agents see things: On visual representations in an emergent language game" (Bouchacourt and Baroni, EMNLP 2018) (https://aclanthology.coli.uni-saarland.de/papers/D18-1119/d18-1119). The paper is based on the signaling game of "Multi-agent cooperation and
the emergence of (natural) language." (Lazaridou et al., ICLR 2017) which is re-implemented here in Pytorch. 

## Create the training and testing dataset

You can create your own versions of the features (from your set of images), this needs to run ```create_features``` with the appropriate parameters ```partition``` and ```sftmax```, and the correct paths in ```produce_vgg_features()``` in ```imagenet_data.py```.

Contact dianeb@fb.com for our pre-processed data (too big to share on github, download link to come soon).

## Create validation dataset (once)
To create the validation dataset once for all, comment ```train()``` and comment out ```create_validation()``` in train.py, then run
```python train.py --manualSeed 0 --same 0``` for the setting where the pair of images are of different concepts (always used in our paper). This creates a fixed validation dataset of images, one for the Sender and one for the Receiver. When training with the same images game (```--noise 0```), the validation images are the ones indexed in ```val_images_indexes_sender``` for both agents, when training with the different images game (```--noise 1```), the images are the ones indexed in ```val_images_indexes_sender``` for the Sender and ```val_images_indexes_receiver``` for the Receiver.

## Run the code 
Then comment out ```train()``` and delete ```create_validation()``` in train.py and run:
```
python train.py --add_one 1 --probs 0 --vocab_size 100 --n_games 50000 --tau_s 10 --grad_clip 1 \
--lr_decay_start 10000 --opti adam --noise 0
```

Note, the model checkpointing every 100 epochs in train.py (see below) can quickly overload the storage capacity, feel free to save models more rarely.
```
  # save current model
  model_save_name = os.path.join(opt.outf,'players' +
                          suffix + '_i%d.pt'%i_games)
  torch.save(players.state_dict(), model_save_name)
```

## Parameters
```
--root # data root folder
--workers # number of data loading workers
--imageSize # the height / width of the input image to VGG network
--lr # learning rate
--lr_decay_start' # learning rate decay iteration start
--lr_decay_every # every how many iter thereafter to div LR by 2
--opti # optimizer
--beta1 # beta1 for adam
--beta2 # beta2 for adam
--cuda # enables cuda
--ngpu # number of GPUs to use
--outf # folder to output images and model checkpoints
--manualSeed # manual seed
--eps # eps for numerical stability
--tau_s # Sender Gibbs temperature
--tau_r # Receiver Gibbs temperature
--game_size # game size
--probs # use softmax layer
--ours # use our data
--add_one # Add 1 to baseline bias
--same # use same concepts
--norm # normalising features
--feat_size # number of image features
--vocab_size #vocabulary size
--batch_size # batch size
--embedding_size# embedding size
--hidden_size # hidden size (number of filters informed sender)
--n_games # number of games
--val_images_use # number of val images to use (set to 1000 gives 1024 because of batch size)
--grad_clip # gradient clipping
--epoch_test # epoch for testing
--noise # If 0, agents see the same images
--inf_rec # Use informed receiver
