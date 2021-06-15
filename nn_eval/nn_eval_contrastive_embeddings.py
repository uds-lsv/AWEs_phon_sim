#!/usr/bin/env python
# coding: utf-8
import os
import sys

import os,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import yaml
import pprint
import collections
import pickle
import matplotlib.pyplot as plt

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


# # make a model str ID, this will be used to save model on desk
# config_args['model_str'] = '_'.join(str(_var) for _var in
#     [
#         current_time_str,
#         config_args['experiment_name'],
#         config_args['input_signal_params']['acoustic_features']
#     ]
# )


# make the dir str where the model will be stored
if config_args['expand_filepaths_to_save_dir']:
    config_args['model_state_file'] = os.path.join(
        config_args['model_save_dir'], config_args['pretrained_model']
    )

    print("Expanded filepaths: ")
    print("\t{}".format(config_args['model_state_file']))

# # if dir does not exits on desk, make it
# train_utils.handle_dirs(config_args['model_save_dir'])


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


#speech_df = speech_df.sample(n=500, random_state=1)

# shuffle splits among words
#speech_df['split'] = np.random.permutation(speech_df.split)

word_vocab = set(speech_df['correct_IPA'].values)

word_featurizer = WordFeaturizer(
    data_dir=config_args['data_dir'],
    acoustic_features= config_args['input_signal_params']['acoustic_features'],
    max_num_frames=config_args['input_signal_params']['max_num_frames'],
    spectral_dim=config_args['acoustic_encoder']['spectral_dim'],
    word_vocab=word_vocab
)

print('WordFeaturizer was initialized:', word_featurizer.max_symbol_sequence_len)

#  dataloader ...
word_dataset = WordPairDataset(speech_df, word_featurizer)


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



word_encoder = WordPairEncoder(acoustic_encoder, 
    encoder_type=config_args['acoustic_encoder']['encoder_arch']
)


word_encoder.load_state_dict(torch.load(config_args['model_state_file']))
# move model to GPU
word_encoder = word_encoder.cuda()
word_encoder.eval()

print(word_encoder)

#word_encoder.apply(train_utils.weights_init_uniform)

loss_func = nn.BCEWithLogitsLoss(reduction='sum')


#ngram_weight_vector = word_featurizer.ngram_weight_vector.cuda()

batch_size = config_args['batch_size']

word_dataset.set_mode(config_args['eval_split'])

AP_scores = []

print('Evaluation started ...')


#segment2vec = collections.defaultdict()

try:
    ### eval ...
    # run one validation pass over the validation split

    batch_generator = generate_batches(word_dataset,
        batch_size=batch_size, device=config_args['device'], 
        drop_last_batch=False,shuffle_batches=False
    )
    num_batches = word_dataset.get_num_batches(batch_size)

    val_running_loss = 0.0

    for batch_index, val_batch_dict in enumerate(batch_generator):


        # forward pass, get embeddings 
        acoustic_embs, _ = word_encoder(
            acoustic_ank=val_batch_dict['anchor_acoustic_word'], 
            acoustic_pos=val_batch_dict['positive_acoustic_word'], 
            device=config_args['device']
        )
        


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
            word_orth_list = [w for w in words_in_batch]
        else:
            acoustic_vectors = np.concatenate(
                (acoustic_vectors, acoustic_embs.cpu().detach().numpy()),
                axis=0
            )
            word_labels_list.extend(word_labels)
            word_orth_list.extend([w for w in words_in_batch])


        print(f"{config_args['pretrained_model']}    "
            f"[{batch_index + 1:>4}/{num_batches:>4}]   "
        )

    #print(word_labels_list[:10])

    print(f"Size of validation set: {len(acoustic_vectors)}")
    # at the end of one validation block, compute AP


    # open word distance dictionary

    with open(config_args['word_dist_dic'], "rb") as input_file:
        word2dist = pickle.load(input_file)

    print(acoustic_vectors[0])
    print(acoustic_vectors[1])


    # MAP
    acoustic_mAP = train_utils.average_precision(
        acoustic_vectors,
        acoustic_vectors,
        word_labels_list,
        word_labels_list,
        label2word,
        single_view=True
    )

    print(
        f"L: {len(acoustic_mAP)}    "
        f"acoustic mAP: {np.mean(acoustic_mAP):<1.3f}    "
        f"median AP: {np.median(acoustic_mAP):<1.3f}    "
        f"std AP: {np.std(acoustic_mAP):<1.3f}    "
        f"min AP: {np.min(acoustic_mAP):<1.3f}    "
        f"max AP: {np.max(acoustic_mAP):<1.3f}    "
    )

    
    plt.figure(figsize=(10,5))
    plt.grid(True)
    plt.hist(acoustic_mAP, density=False, bins=20, color='g'); # density=False would make counts
    plt.axvline(x=np.mean(acoustic_mAP), color='r');
    plt.xticks([(x/10) for x in range(0, 11, 1)]);
    plt.xlabel('mAP', fontsize=18)
    plt.xlim(0,1)

    plt.savefig(config_args['save_fig'] + 'mAP.pdf')

    ### TAU

    index2word = {idx: word for idx, word in enumerate(word_orth_list)}

    tau_values, spr_values = train_utils.compute_kendalls_tau(
        acoustic_vectors,
        acoustic_vectors,
        word2dist,
        index2word
    )

    mean_tau = np.mean(tau_values)
    mean_spr = np.mean(spr_values)

    print(
        f"L: {len(tau_values)}    "
        f"mean tau: {mean_tau:<1.3f}    "
        f"median tau: {np.median(tau_values):<1.3f}    "
        f"std tau: {np.std(tau_values):<1.3f}    "
        f"min tau: {np.min(tau_values):<1.3f}    "
        f"max tau: {np.max(tau_values):<1.3f}    "
    )

    print(
        f"L: {len(spr_values)}    "
        f"mean spr: {mean_spr:<1.3f}    "
        f"median spr: {np.median(spr_values):<1.3f}    "
        f"std spr: {np.std(spr_values):<1.3f}    "
        f"min spr: {np.min(spr_values):<1.3f}    "
        f"max spr: {np.max(spr_values):<1.3f}    "
    )

    
    plt.figure(figsize=(10,2))
    plt.grid(True)
    plt.hist(tau_values, density=False, bins=20, color='g'); # density=False would make counts
    plt.axvline(x=np.mean(tau_values), color='r');
    plt.xticks([(x/10) for x in range(-10, 11, 2)]);
    plt.xlabel('Tau', fontsize=18)
    plt.xlim(-1,1)

    plt.savefig(config_args['save_fig'] + 'tau.pdf')

except KeyboardInterrupt:
    print("Exiting loop")
