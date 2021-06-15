#!/usr/bin/env python
# coding: utf-8
import os
import sys
import yaml
import pprint
import collections

# to get time model was trained
from datetime import datetime
import pytz

# NOTE: import torch before pandas, otherwise segementation fault error occurs
# The couse of this problem is UNKNOWN, and not solved yet
import torch
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.optim as optim

from nn_speech_models import *
import train_utils

import torch.nn.functional as F


# obtain yml config file from cmd line and print out content
if len(sys.argv) != 2:
	sys.exit("\nUsage: " + sys.argv[0] + " <config YAML file>\n")

config_file_path = sys.argv[1] # e.g., '/speech_cls/config_1.yml'
config_args = yaml.safe_load(open(config_file_path))
print('YML configuration file content:')
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(config_args)


# get time in CET timezone
current_time = datetime.now(pytz.timezone('Europe/Amsterdam'))
current_time_str = current_time.strftime("%d%m%Y_%H_%M_%S") # YYYYMMDD HH:mm:ss
#print(current_time_str)


# make a model str ID, this will be used to save model on desk
config_args['model_str'] = '_'.join(str(_var) for _var in
    [
        current_time_str,
        config_args['experiment_name'],
        config_args['input_signal_params']['acoustic_features']
    ]
)


# make the dir str where the model will be stored
if config_args['expand_filepaths_to_save_dir']:
    config_args['model_state_file'] = os.path.join(
        config_args['model_save_dir'], config_args['model_str']
    )

    print("Expanded filepaths: ")
    print("\t{}".format(config_args['model_state_file']))

# if dir does not exits on desk, make it
train_utils.handle_dirs(config_args['model_save_dir'])


 # Check CUDA
if not torch.cuda.is_available():
    config_args['cuda'] = False

config_args['device'] = torch.device("cuda" if config_args['cuda'] else "cpu")

print("Using CUDA: {}".format(config_args['cuda']))


# Set seed for reproducibility
train_utils.set_seed_everywhere(config_args['seed'], config_args['cuda'])




##### HERE IT ALL STARTS ...
# dataset  & featurizer ...
speech_df = pd.read_csv(config_args['speech_metadata'],
    delimiter="\t", encoding='utf-8')


label_set=config_args['language_set'].split()


speech_df = speech_df[
    (speech_df.language.isin(label_set)) &
    (speech_df.num_ph>3) &
    (speech_df.frequency>1) &
    (speech_df.duration<1.10)
]

#print(speech_df.head())
print(len(speech_df))


#speech_df = speech_df.sample(n=5000, random_state=1)

# shuffle splits among words
#speech_df['split'] = np.random.permutation(speech_df.split)

word_vocab = set(speech_df['correct_IPA'].values)

word_featurizer = WordNgramFeaturizer(
    data_dir=config_args['data_dir'],
    acoustic_features= config_args['input_signal_params']['acoustic_features'],
    max_num_frames=config_args['input_signal_params']['max_num_frames'],
    spectral_dim=config_args['acoustic_encoder']['spectral_dim'],
    word_vocab=word_vocab
)

print('WordFeaturizer was initialized with ngram ', word_featurizer.ngram_vector_size)

# print ngrams 
# for ngram, i in word_featurizer.ngram2index.items():
#     print(i, ngram.encode('utf-8'))


#  dataloader ...
word_dataset = WordDataset(speech_df, word_featurizer, ngram_featurizer=True)


word2label = {word:idx for idx, word in enumerate(set(speech_df.orth.values))}
label2word = {idx:word for word, idx in word2label.items()}



# initialize acoustic encoder
if config_args['acoustic_encoder']['encoder_arch']=='CNN':
    # initialize a CNN encoder
    acoustic_encoder = AcousticEncoder(
        spectral_dim=config_args['acoustic_encoder']['spectral_dim'],
        max_num_frames=config_args['acoustic_encoder']['max_num_frames'],
        output_dim=config_args['acoustic_encoder']['output_dim'],
        num_channels=config_args['acoustic_encoder']['num_channels'],
        filter_sizes=config_args['acoustic_encoder']['filter_sizes'],
        stride_steps=config_args['acoustic_encoder']['stride_steps'],
        pooling_type=config_args['acoustic_encoder']['pooling_type']
    )

elif config_args['acoustic_encoder']['encoder_arch']=='LSTM':
    # initialize an LSTM encoder
    acoustic_encoder = AcousticEncoderLSTM(
        spectral_dim=config_args['acoustic_encoder']['spectral_dim'],
        max_num_frames=config_args['acoustic_encoder']['max_num_frames'],
        output_dim=config_args['acoustic_encoder']['output_dim'],
        hidden_state_dim=config_args['acoustic_encoder']['hidden_state_dim'],
        n_layers=config_args['acoustic_encoder']['n_layers'],
        unit_dropout_prob=config_args['acoustic_encoder']['unit_dropout_prob']
    )

elif config_args['acoustic_encoder']['encoder_arch']=='BiLSTM':
    # initialize an LSTM encoder
    acoustic_encoder = AcousticEncoderBiLSTM(
        spectral_dim=config_args['acoustic_encoder']['spectral_dim'],
        max_num_frames=config_args['acoustic_encoder']['max_num_frames'],
        output_dim=config_args['acoustic_encoder']['output_dim'],
        hidden_state_dim=config_args['acoustic_encoder']['hidden_state_dim'],
        n_layers=config_args['acoustic_encoder']['n_layers'],
        unit_dropout_prob=config_args['acoustic_encoder']['unit_dropout_prob']
    )

elif config_args['acoustic_encoder']['encoder_arch']=='BiGRU':
    # initialize an LSTM encoder
    acoustic_encoder = AcousticEncoderBiGRU(
        spectral_dim=config_args['acoustic_encoder']['spectral_dim'],
        max_num_frames=config_args['acoustic_encoder']['max_num_frames'],
        output_dim=config_args['acoustic_encoder']['output_dim'],
        hidden_state_dim=config_args['acoustic_encoder']['hidden_state_dim'],
        n_layers=config_args['acoustic_encoder']['n_layers'],
        unit_dropout_prob=config_args['acoustic_encoder']['unit_dropout_prob']
    )

else:
    raise NotImplementedError


word_encoder = CLSpeechEncoder(
    acoustic_encoder, 
    hidden_dim=config_args['acoustic_encoder']['output_dim'], 
    num_classes= word_featurizer.ngram_vector_size
)

print(word_encoder)

word_encoder.apply(train_utils.weights_init_uniform)

# move model to GPU
word_encoder = word_encoder.cuda()


loss_func = nn.BCEWithLogitsLoss(reduction='sum')

optimizer = optim.Adam(
    word_encoder.parameters(),
    lr=config_args['training_hyperparams']['learning_rate']
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode='max',
    factor=0.5,
    patience=10
)

train_state = train_utils.make_train_state(config_args)
ngram_weight_vector = word_featurizer.ngram_weight_vector.cuda()

num_epochs = config_args['training_hyperparams']['num_epochs']
batch_size = config_args['training_hyperparams']['batch_size']


AP_scores = []

print('Training started ...')

try:
    # iterate over training epochs ...
    for epoch_index in range(num_epochs):
        ### TRAINING ...
        train_state['epoch_index'] = epoch_index

        # Iterate over training dataset, set loss and acc to 0
        # set train mode on, generate batch

        running_loss = 0.0

        word_encoder.train()

        word_dataset.set_mode('TRA')

        batch_generator = generate_batches(word_dataset,
            batch_size=batch_size, device=config_args['device']
        )

        num_batches = word_dataset.get_num_batches(batch_size)

        # iterate over training batches
        for batch_index, batch_dict in enumerate(batch_generator):

            # zero the gradients
            optimizer.zero_grad()

            # forward pass, get logits 
            y_hat = word_encoder(
                acoustic_in=batch_dict['acoustic_word'], 
                device=config_args['device'],
                encoder_type=config_args['acoustic_encoder']['encoder_arch']
            )

            y_tar = batch_dict['symbolic_word']

            #print(y_tar)

            
            loss = loss_func(y_hat, y_tar)

            # cost sensetive learning
            #print(loss.shape)
            #loss = (loss * ngram_weight_vector).mean()

            loss.backward() # backprop

            # `clip_grad_norm` helps prevent the exploding gradient problem in LSTMs.
            nn.utils.clip_grad_norm_(word_encoder.parameters(), 0.25)

            optimizer.step() # optimizer step

            # compute running loss
            batch_loss = loss.item()
            running_loss += (batch_loss - running_loss)/(batch_index + 1)

            print(f"{config_args['model_str']}    "
                f"TRA epoch [{epoch_index + 1:>2}/{num_epochs}]"
                f"[{batch_index + 1:>4}/{num_batches:>4}]   "
                f"b-loss: {batch_loss:1.4f}   "
                f"r-loss: {running_loss:1.4f}   "
                #f"lr: {scheduler.get_lr()[0]:0.8f}"
            )


        # one training epoch is DONE! Update training state
        train_state['train_loss'].append(running_loss)

        ### VALIDATION ...
        # run one validation pass over the validation split
        word_encoder.eval()

        word_dataset.set_mode(config_args['dev_split'])

        batch_generator = generate_batches(word_dataset,
            batch_size=batch_size, device=config_args['device'], 
            drop_last_batch=False,shuffle_batches=False
        )

        num_batches = word_dataset.get_num_batches(batch_size)

        val_running_loss = 0.0

        for batch_index, val_batch_dict in enumerate(batch_generator):
            # forward pass, get embeddings from both views
            y_hat, acoustic_embs = word_encoder(
                acoustic_in=val_batch_dict['acoustic_word'], 
                encoder_type=config_args['acoustic_encoder']['encoder_arch'],
                device=config_args['device'],
                return_vector=True
            )

            #print(acoustic_embs.shape)
            y_tar = val_batch_dict['symbolic_word']

            # get word orthographic form for all word in batch
            words_in_batch = val_batch_dict['orth_sequence']

            #print(words_in_batch)

            # make word labels, this makes it possible to know whether or
            # not the representation belong to the same word in each view
            word_labels = [word2label[w] for w in words_in_batch]

            #rand_idx = torch.randint(0, acoustic_embs.shape[0], (acoustic_embs.shape[0],))

            # get vectors
            if batch_index == 0:
                acoustic_vectors = acoustic_embs.cpu().detach().numpy()
                word_labels_list = word_labels
            else:
                acoustic_vectors = np.concatenate(
                    (acoustic_vectors, acoustic_embs.cpu().detach().numpy()),
                    axis=0
                )
                word_labels_list.extend(word_labels)

            # added y_hat[:len(y_tar), :] to account for the last val iteration 
            # which less samples than the batch size
            val_loss = loss_func(y_hat, y_tar)

            #val_loss = (val_loss * ngram_weight_vector).mean()

             # compute running loss
            batch_loss = val_loss.item()
            val_running_loss += (batch_loss - val_running_loss)/(batch_index + 1)

            print(f"{config_args['model_str']}    "
                f"VAL epoch [{epoch_index + 1:>2}/{num_epochs}]"
                f"[{batch_index + 1:>4}/{num_batches:>4}]   "
                f"b-loss: {batch_loss:1.4f}   "
                f"r-loss: {val_running_loss:1.4f} "
                #f"acc: {run_cls_acc:2.2f}"
            )

        #print(word_labels_list[:10])

        print(f"Size of validation set: {len(acoustic_vectors)}")
        # at the end of one validation block, compute AP


        print(acoustic_vectors[0])
        print(acoustic_vectors[1])
        acoustic_mAP = train_utils.average_precision(
            acoustic_vectors,
            acoustic_vectors,
            word_labels_list,
            word_labels_list,
            label2word,
            single_view=True
        )

        mean_acoustic_mAP = np.mean(acoustic_mAP)

        AP_scores.append(mean_acoustic_mAP)

        print(
            f"Validation results epoch {epoch_index +1}: "
            f"L: {len(acoustic_mAP)} "
            f"acoustic AP: {mean_acoustic_mAP:<1.6f}"
        )

        # TRAIN & VAL iterations for one epoch is over ...
        train_state['val_loss'].append(val_running_loss)
        train_state['val_acoustic_mAP'].append(acoustic_mAP)
        #train_state['val_crossview_mAP'].append(cross_mAP)

        train_state = train_utils.update_train_state(args=config_args,
            model=word_encoder,
            train_state=train_state
        )

        scheduler.step(train_state['val_loss'][-1])
        #scheduler.step()

except KeyboardInterrupt:
    print("Exiting loop")


# once training is over for the number of batches specified, check best epoch
for i, aAP in enumerate(AP_scores):
    #print(aAP, xAP)
    print("Validation performance @ epoch {}: acoustic AP {:.5f}".format(i+1, aAP))


print('Best {} model by acoustic AP: {:.5f} epoch {}'.format(
	config_args['model_str'],
	max(AP_scores),
    1 + np.argmax(AP_scores))
)
