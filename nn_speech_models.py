# coding: utf-8
# Code for Acoustic Word Representations (AWEs)
# Developed by Badr M. Abdullah — LSV @ Saarland University
# Follow me on Twitter @badr_nlp

import math
from collections import defaultdict, Counter

import sklearn
import sklearn.preprocessing

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Function


##### CLASS Vocabulary: A class to represent vacabulary (set of symbols)
class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping. """
    def __init__(self, symbol2idx=None):
        """
        Args: symbol2idx (dict): a pre-existing map of symbols to indices
        """
        if symbol2idx is None:
            symbol2idx = {}

        self._symbol2idx = symbol2idx

        self._idx2symbol = {
            idx: symbol for symbol, idx in self._symbol2idx.items()
        }


    def add_symbol(self, symbol):
        """Update mapping dicts based on the symbol.
        Args: symbol (str): the item to add into the Vocabulary
        Returns: index (int): the integer corresponding to the symbol
        """
        if symbol in self._symbol2idx:
            index = self._symbol2idx[symbol]

        else:
            index = len(self._symbol2idx)
            self._symbol2idx[symbol] = index
            self._idx2symbol[index] = symbol

        return index


    def add_many(self, symbols):
        """Add a list of symbols into the Vocabulary
        Args: symbols (list): a list of string symbols
        Returns: indices (list): a list of indices corresponding to the symbols
        """
        return [self.add_symbol(symbol) for symbol in symbols]


    def lookup_symbol(self, symbol):
        """Retrieve the index associated with the symbol

        Args: symbol (str): the symbol to look up
        Returns: index (int): the index corresponding to the symbol
        """
        return self._symbol2idx[symbol]


    def lookup_index(self, index):
        """Return the symbol associated with the index

        Args: index (int): the index to look up
        Returns: symbol (str): the symbol corresponding to the index
        Raises: KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx2symbol:
            raise KeyError(f"the index {index} is not in the Vocabulary." )

        return self._idx2symbol[index]


    def __str__(self):
        return f"<Vocabulary(size={len(self)})>"

    def __len__(self):
        return len(self._symbol2idx)


##### CLASS SequenceVocabulary: A class to represent vacabulary (set of symbols)
##### but for tasks that require sequence processing (e.g padding to fixed len)
class SequenceVocabulary(Vocabulary):
    """ """
    def __init__(self,
        symbol2idx=None,
        unk_symbol="<UNK>",
        mask_symbol="<MASK>",
        begin_seq_symbol="<BEGIN>",
        end_seq_symbol="<END>"
    ):

        super(SequenceVocabulary, self).__init__(symbol2idx)

        self._mask_symbol = mask_symbol
        self._unk_symbol = unk_symbol
        self._begin_seq_symbol = begin_seq_symbol
        self._end_seq_symbol = end_seq_symbol

        self.mask_index = self.add_symbol(self._mask_symbol)
        self.unk_index = self.add_symbol(self._unk_symbol)
        self.begin_seq_index = self.add_symbol(self._begin_seq_symbol)
        self.end_seq_index = self.add_symbol(self._end_seq_symbol)



    def lookup_symbol(self, symbol):
        """Retrieve the index associated with the symbol
          or the UNK index if symbol isn't present.

        Args: symbol (str): the symbol to look up
        Returns: index (int): the index corresponding to the symbol
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality
        """
        if self.unk_index >= 0:
            return self._symbol2idx.get(symbol, self.unk_index)
        else:
            return self._symbol2idx[symbol]


##### CLASS WordFeaturizer: A featurizer for acoustic word embeddings
class WordFeaturizer(object):
    """ The Featurizer handles the low-level speech features and labels. """

    def __init__(self,
        data_dir,
        acoustic_features,
        max_num_frames,
        spectral_dim,
        word_vocab
    ):
        """
        Args:
            data_dir (str): the path to the data on disk to read .npy files
            acoustic_features (str): low-level speech features, e.g., MFCCs
            max_num_frames (int): the max number of acoustic frames in input
                the diff. (max_num_frames - num_frames) is padded with zeros
            spectral_dim (int): the num of spectral components (default 13)
        """
        self.data_dir = data_dir
        self.acoustic_features = acoustic_features
        self.max_num_frames = max_num_frames
        self.spectral_dim = spectral_dim
        self.word_vocab = word_vocab

        # make char vocab
        all_chars = [c for w in self.word_vocab for c in w]
        #self.char_set = set(all_chars)

        char_counter = Counter(all_chars)

        max_word_len = max(map(len, self.word_vocab))

        self.char_vocab = SequenceVocabulary()

        for char in char_counter:
            self.char_vocab.add_symbol(char)


        self.max_symbol_sequence_len = max(map(len, self.word_vocab)) + 2

    def get_acoustic_form(self,
        segment_ID,
        segment_start,
        segment_end,
        max_num_frames=None,
        spectral_dim=None
    ):
        """
        Given a segment ID and other variables of spectral features,
        return a low-level spectral representation of word segment (e.g., MFCCs)
        Args:
            segment_ID (str): segment ID, i.e., the ID of word segment
            max_num_frames (int): max length of the (MFCC) vector sequence
            spectral_dim (int): the num of spectral coefficient (default 13)

        Returns:
            low-level speech representation as a PyTorch Tensor
            (torch.Tensor: max_num_frames x spectral_dim)
        """
        # these were added to enable differet uttr lengths during inference
        if max_num_frames is None: max_num_frames = self.max_num_frames
        if spectral_dim is None: spectral_dim = self.spectral_dim

        # path to feature matrix (e.g. MFCCs saved in .npy file)
        file_name = self.data_dir + segment_ID + '.' + \
            self.acoustic_features.lower() + '.npy' #

        # load feature representation from desk
        _start_frame = int(segment_start*100)
        _end_frame   = int(segment_end*100)
        acoustic_segment = np.load(file_name)[:, _start_frame:_end_frame]

        # scale feuture vector to have zero mean and unit var
        acoustic_segment = sklearn.preprocessing.scale(acoustic_segment, axis=1)

        # get word length in num of frames
        segment_len = acoustic_segment.shape[1]

        # assert that the word segment length is less than max num frames
        assert segment_len < max_num_frames, \
            f"Error: max num of frames error {segment_len}, {max_num_frames}"

        # convert to pytorch tensor
        acoustic_tensor = torch.from_numpy(acoustic_segment)

        # apply padding to the segment represenation
        acoustic_tensor_pad = torch.zeros(spectral_dim, max_num_frames)

        # sample a random start index
        _start_idx = torch.randint(1 + max_num_frames - segment_len, (1,)).item()


        acoustic_tensor_pad[:spectral_dim,_start_idx:_start_idx + segment_len] = \
            acoustic_tensor[:spectral_dim,:segment_len]


        return acoustic_tensor_pad.float() # convert to float tensor

    def get_symbolic_form(self,
        symbol_sequence,
        #vector_length=-1,
        add_language_tok=False
    ):
        """
        Given a segment ID and flag for language token embedding
        return the symbolic sequence of the word form
        Args:
            segment_ID (str): segment ID, i.e., the ID of word segment
            add_language_tok (bool): whether or not to add language token

        Returns:
            char sequence
            (torch.Tensor: max_num_frames x spectral_dim)
        """

        # add first symbol BEGIN
        indices = [self.char_vocab.begin_seq_index]

        # add other tokens
        indices.extend(
            self.char_vocab.lookup_symbol(char) for char in symbol_sequence
        )

        # add last symbol END
        indices.append(self.char_vocab.end_seq_index)

        #if vector_length < 0:
        #   vector_length = len(indices)

        index_sequence = np.zeros(self.max_symbol_sequence_len, dtype=np.int64)
        index_sequence[:len(indices)] = indices
        index_sequence[len(indices):] = self.char_vocab.mask_index

        return index_sequence


##### CLASS WordNgramFeaturizer: A featurizer for acoustic word embeddings w/ ngrams
class WordNgramFeaturizer(object):
    """ The Featurizer handles the low-level speech features and symbol ngrams. """

    def __init__(self,
        data_dir,
        acoustic_features,
        max_num_frames,
        spectral_dim,
        word_vocab
    ):
        """
        Args:
            data_dir (str): the path to the data on disk to read .npy files
            acoustic_features (str): low-level speech features, e.g., MFCCs
            max_num_frames (int): the max number of acoustic frames in input
                the diff. (max_num_frames - num_frames) is padded with zeros
            spectral_dim (int): the num of spectral components (default 13)
        """
        self.data_dir = data_dir
        self.acoustic_features = acoustic_features
        self.max_num_frames = max_num_frames
        self.spectral_dim = spectral_dim
        self.word_vocab = word_vocab

        all_ngrams = [
            ngram for w in word_vocab for ngram in self.get_ngrams(w)
        ]

        self.ngram_counter = Counter(all_ngrams)

        self.ngram_set = [
            g for g, c in self.ngram_counter.most_common()
            if self.ngram_counter[g] > 100
        ]

        self.ngram_weight_vector = self.get_ngram_weight_vector()

        self.ngram2index = {g: i for i, g in enumerate(self.ngram_set)}
        self.index2ngram = {i: g for g, i in self.ngram2index.items()}

        self.ngram_vector_size = len(self.ngram2index)


    def get_ngram_weight_vector(self):
        """Return weight vecotr for the ngrams"""

        ngram_weight_vector = [
            500/c for g, c in self.ngram_counter.most_common(len(self.ngram_set))
        ]

        return torch.tensor(ngram_weight_vector) #, dtype=torch.float32


    def get_ngrams(self, word):
        """Get a word (str), return a list of ngrams in the word.
        Example:
            >>> get_ngrams('kaže')
            ['k', 'a', 'ž', 'e', '#k', 'ka', 'až', 'že', 'e#', '#ka', 'kaž', 'aže', 'že#']

        """
        #print(word.encode('utf-8'))
        word = ['#'] + word.split() + ['#'] 
        # extract unigrams
        #n_1grams = [word[i - 1] for i in range(1, len(word) + 1)]

        # extract bigrams & trigrams
        # append pseudo-tokens to the beginning & end of the word
        #word = '#' + word + '#'
        n_2grams = [''.join([word[i - 1], word[i]]) for i in range(1, len(word))]
        n_3grams = [''.join([word[i - 2], word[i - 1], word[i]]) for i in range(2, len(word))]

        ngrams = n_2grams + n_3grams #n_1grams+ n_3grams
        

        return ngrams


    def get_ngram_set(self,
        symbol_sequence,
        return_index=True
    ):
        """
        Given a symbol sequence, return ngrams in the symbol sequence up to n=3.
        Args:
            segment_ID (str): segment ID, i.e., the ID of word segment

        Returns:
            ngrams (set)
        """

        assert len(symbol_sequence) > 0, "Word should have at least one char."

        symbol_ngrams = self.get_ngrams(symbol_sequence)

        item_ngram_set = set(c for c in symbol_ngrams if c in self.ngram_set)

        if return_index:
            return {self.ngram2index[g] for g in item_ngram_set}
        else:
            return item_ngram_set


    def vectorize_ngram_set(self, symbol_sequence):
        """Get Given a symbol sequence, return ngrams as a binary vector"""

        ngram_set_by_index = self.get_ngram_set(symbol_sequence)
        ngram_vector = [
            1 if i in ngram_set_by_index else 0
            for i in range(0, self.ngram_vector_size)
        ]

        return torch.tensor(ngram_vector, dtype=torch.float32)#.unsqueeze(dim=0)


    def get_acoustic_form(self,
        segment_ID,
        segment_start,
        segment_end,
        max_num_frames=None,
        spectral_dim=None
    ):
        """
        Given a segment ID and other variables of spectral features,
        return a low-level spectral representation of word segment (e.g., MFCCs)
        Args:
            segment_ID (str): segment ID, i.e., the ID of word segment
            max_num_frames (int): max length of the (MFCC) vector sequence
            spectral_dim (int): the num of spectral coefficient (default 13)

        Returns:
            low-level speech representation as a PyTorch Tensor
            (torch.Tensor: max_num_frames x spectral_dim)
        """
        # these were added to enable differet uttr lengths during inference
        if max_num_frames is None: max_num_frames = self.max_num_frames
        if spectral_dim is None: spectral_dim = self.spectral_dim

        # path to feature matrix (e.g. MFCCs saved in .npy file)
        file_name = self.data_dir + segment_ID + '.' + \
            self.acoustic_features.lower() + '.npy' #
            #self.acoustic_features.lower() + '.norm.npy' #

        # load feature representation from desk
        _start_frame = int(segment_start*100)
        _end_frame   = int(segment_end*100)
        acoustic_segment = np.load(file_name)[:, _start_frame:_end_frame]

        # scale feuture vector to have zero mean and unit var
        acoustic_segment = sklearn.preprocessing.scale(acoustic_segment, axis=1)

        #print(acoustic_segment[:, :1])

        # get word length in num of frames
        segment_len = acoustic_segment.shape[1]

        # assert that the word segment length is less than max num frames
        assert segment_len < max_num_frames, \
            f"Error: max num of frames error {segment_len}, {max_num_frames}"

        # convert to pytorch tensor
        acoustic_tensor = torch.from_numpy(acoustic_segment)

        #print(acoustic_tensor[:, :1])

        # apply padding to the segment represenation
        acoustic_tensor_pad = torch.zeros(spectral_dim, max_num_frames)

        # sample a random start index
        _start_idx = torch.randint(1 + max_num_frames - segment_len, (1,)).item()


        acoustic_tensor_pad[:spectral_dim,_start_idx:_start_idx + segment_len] = \
            acoustic_tensor[:spectral_dim,:segment_len]

        #print(acoustic_tensor_pad[:, _start_idx:_start_idx + 1])


        return acoustic_tensor_pad#.float() # convert to float tensor


##### CLASS WordDataset: A class to handle (batch) data transformation
class WordDataset(Dataset):
    def __init__(self, dataset_df, featurizer, ngram_featurizer=False):
        """
        Args:
            dataset_df (pandas.df): a pandas dataframe (label, split, file)
            featurizer (SpeechFeaturizer): the speech featurizer
        """
        self.dataset_df = dataset_df
        self._word_featurizer = featurizer

        # read data and make splits
        self.train_df = self.dataset_df[self.dataset_df.split=='TRA']
        self.train_size = len(self.train_df)

        self.val_df = self.dataset_df[self.dataset_df.split=='DEV']
        self.val_size = len(self.val_df)

        self.test_df = self.dataset_df[self.dataset_df.split=='EVA']
        self.test_size = len(self.test_df)

        print('Size of the splits (train, val, test): ',  \
             self.train_size, self.val_size, self.test_size)

        self._lookup_dict = {
            'TRA': (self.train_df, self.train_size),
            'DEV': (self.val_df, self.val_size),
            'EVA': (self.test_df, self.test_size)
        }

        # by default set mode to train
        self.set_mode(split='TRA')


        self.vectorize_ngrams = ngram_featurizer


    def set_mode(self, split='TRA'):
         """Set the mode using the split column in the dataframe. """
         self._target_split = split
         self._target_df, self._target_size = self._lookup_dict[split]


    def __len__(self):
        "Returns the number of the data points in the target split."
        return self._target_size


    def __getitem__(self, index):
        """
        A data transformation code for one data point in the dataset.
        Args:
            index (int): the index to the data point in the target dataframe
        Returns:
            a dictionary holding the point representation, e.g.,
                acoustic view (x_acoustic), orth view (x_orth)
        """

        speech_segment = self._target_df.iloc[index]

        # get acoustic form of the word
        acoustic_word = self._word_featurizer.get_acoustic_form(
            segment_ID=speech_segment.seg_id,
            segment_start=speech_segment.start,
            segment_end=speech_segment.end
        )

        # get symbolic form of the word
        if self.vectorize_ngrams:
            symbolic_word = self._word_featurizer.vectorize_ngram_set(
                #speech_segment.orth
                speech_segment.correct_IPA
            )


        else:
            symbolic_word = self._word_featurizer.get_symbolic_form(
                speech_segment.correct_IPA
            )

        return {
            'acoustic_word': acoustic_word,
            'symbolic_word': symbolic_word,
            'orth_sequence': speech_segment.orth,
            'word_seg_id': speech_segment.word_id
        }

    def get_num_batches(self, batch_size):
        """
        Given batch size (int), return the number of dataset batches (int)
        """
        return math.ceil((len(self) / batch_size))


##### CLASS WordPairDataset: A class to handle (batch) word pair transformation
class WordPairDataset(Dataset):
    def __init__(self, dataset_df, featurizer):
        """
        Args:
            dataset_df (pandas.df): a pandas dataframe (label, split, file)
            featurizer (SpeechFeaturizer): the speech featurizer
        """
        self.dataset_df = dataset_df
        self._word_featurizer = featurizer

        # read data and make splits
        self.train_df = self.dataset_df[self.dataset_df.split=='TRA']
        self.train_size = len(self.train_df)

        self.val_df = self.dataset_df[self.dataset_df.split=='DEV']
        self.val_size = len(self.val_df)

        self.test_df = self.dataset_df[self.dataset_df.split=='EVA']
        self.test_size = len(self.test_df)

        print('Size of the splits (train, val, test): ',  \
             self.train_size, self.val_size, self.test_size)

        self._lookup_dict = {
            'TRA': (self.train_df, self.train_size),
            'DEV': (self.val_df, self.val_size),
            'EVA': (self.test_df, self.test_size)
        }

        # by default set mode to train
        self.set_mode(split='TRA')



    def set_mode(self, split='TRA'):
         """Set the mode using the split column in the dataframe. """
         self._target_split = split
         self._target_df, self._target_size = self._lookup_dict[split]


    def __len__(self):
        "Returns the number of the data points in the target split."
        return self._target_size


    def __getitem__(self, index):
        """
        A data transformation code for one data point in the dataset.
        Args:
            index (int): the index to the data point in the target dataframe
        Returns:
            a dictionary holding the point representation, e.g.,
                acoustic view (x_acoustic), orth view (x_orth)
        """

        speech_segment_ank = self._target_df.iloc[index]

        # get acoustic form of the anchor point
        acoustic_word_ank = self._word_featurizer.get_acoustic_form(
            segment_ID=speech_segment_ank.seg_id,
            segment_start=speech_segment_ank.start,
            segment_end=speech_segment_ank.end
        )

        # get acoustic form of a positive point
        # first sample a row from the target dataframe with same orth word
        condition = (self._target_df.orth==speech_segment_ank.orth)
        speech_segment_pos = self._target_df[condition].sample(n = 1).iloc[0] #, random_state=1
        
        # get acoustic form for the positive point
        acoustic_word_pos = self._word_featurizer.get_acoustic_form(
            segment_ID=speech_segment_pos.seg_id,
            segment_start=speech_segment_pos.start,
            segment_end=speech_segment_pos.end
        )

        return {
            'anchor_acoustic_word': acoustic_word_ank,
            'positive_acoustic_word': acoustic_word_pos,
            'orth_sequence': speech_segment_ank.orth,
            'word_seg_id': speech_segment_ank.word_id
        }



    def get_num_batches(self, batch_size):
        """
        Given batch size (int), return the number of dataset batches (int)
        """
        return math.ceil((len(self) / batch_size))


##### A METHOD TO GENERATE BATCHES WITH A DATALOADER WRAPPER
def generate_batches(word_dataset, batch_size, shuffle_batches=True,
    drop_last_batch=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader and ensures that
      each tensor is on the right device (i.e., CPU or GPU).
    """
    dataloader = DataLoader(dataset=word_dataset, batch_size=batch_size,
        shuffle=shuffle_batches, drop_last=drop_last_batch)

    # for each batch, yield a dictionary with keys: A_word, O_word
    for data_dict in dataloader:
        # an dict object to yield in each iteration
        batch_data_dict = {}

        for name, _ in data_dict.items():
            if name not in ['orth_sequence', 'word_seg_id']:
                batch_data_dict[name] = data_dict[name].to(device)
            else:
                batch_data_dict[name] = data_dict[name]

        yield batch_data_dict

        # for var_key in data_dict:
        #     batch_data_dict[var_key] = data_dict[var_key].to(device)
        #
        # yield batch_data_dict


#### CLASS FrameDropout: A custome layer for frame dropout
class FrameDropout(nn.Module):
    def __init__(self, dropout_prob=0.2):
        """Applies dropout on the frame level so entire feature vector will be
            evaluated to zero vector with probability p.
        Args:
            p (float): dropout probability
        """
        super(FrameDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x_in):
        _, _, sequence_dim = x_in.shape

        # randomly sample frame indecies to be dropped
        drop_frame_idx = [i for i in range(sequence_dim) \
            if torch.rand(1).item() < self.dropout_prob]

        x_in[:, :, drop_frame_idx] = 0

        return x_in


##### CLASS SpectralDropout: A custome layer for spectral (coefficient) dropout
class SpectralDropout(nn.Module):
    def __init__(self, dropout_prob=0.2, feature_idx=None):
        """Applies dropout on the feature level so spectral component accross
             vectors are replaced with zero (row-)vector with probability p.
        Args:
            p (float): dropout probability
            feature_idx (int): to mask specific spectral coeff. during inference
        """
        super(SpectralDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x_in):
        batch_size, spectral_dim, sequence_dim = x_in.shape

        # randomly sample frame indecies to be dropped
        drop_feature_idx = [i for i in range(spectral_dim) \
            if torch.rand(1).item() < self.dropout_prob]

        x_in[:, drop_feature_idx, :] = 0

        return x_in


##### CLASS FrameReverse: A custome layer for frame sequence reversal
class FrameReverse(nn.Module):
    def __init__(self):
        """Reverses the frame sequence in the input signal. """
        super(FrameReverse, self).__init__()

    def forward(self, x_in):
        batch_size, spectral_dim, sequence_dim = x_in.shape
        # reverse indicies
        reversed_idx = [i for i in reversed(range(sequence_dim))]
        x_in[:, :, reversed_idx] = x_in

        return x_in


##### CLASS FrameShuffle: A custome layer for frame sequence shuflle
class FrameShuffle(nn.Module):
    def __init__(self):
        """Shuffle the frame sequence in the input signal, given a bag size. """
        super(FrameShuffle, self).__init__()

    def forward(self, x_in, bag_size=1):
        batch_size, spectral_dim, seq_dim = x_in.shape

        # shuffle idicies according to bag of frames size
        # make the bags of frames
        seq_idx = list(range(seq_dim))

        # here, a list of bags (lists) will be made
        frame_bags = [seq_idx[i:i+bag_size] for i in range(0, seq_dim, bag_size)]

        # shuffle the bags
        random.shuffle(frame_bags)

        # flatten the bags into a sequential list
        shuffled_idx = [idx for bag in frame_bags for idx in bag]

        x_in[:, :, shuffled_idx] = x_in

        return x_in


##### CLASS GaussianNoise: A custome layer for additive Gaussian noise
class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.1):
        super().__init__()
        self.stddev = stddev

    def forward(self, x_in):
        if self.training:
            return x_in + torch.autograd.Variable(torch.randn(x_in.size()).cuda() * self.stddev)
        return x_in


##### CLASS AcousticEncoder: A multi-layer convolutional acoustic encoder
class AcousticEncoder(nn.Module):

    """A a temporal convolutional encoder for speech data."""
    def __init__(self,
        spectral_dim=13,
        max_num_frames= 384,
        num_channels=[128, 256, 512],
        filter_sizes=[8, 12, 16],
        stride_steps=[1, 1, 1],
        output_dim=512,
        pooling_type='max',
        signal_dropout_prob=0.0,
        unit_dropout_prob=0.2, 
        dropout_frames=False,
        dropout_spectral_features=False,
        mask_signal=False
    ):
        """
        Args:
            spectral_dim (int): number of spectral coefficients
            max_num_frames (int): max number of acoustic frames in input
            num_channels (list): number of channels per each Conv layer
            filter_sizes (list): size of filter/kernel per each Conv layer
            stride_steps (list): strides per each Conv layer
            pooling (str): pooling procedure, either 'max' or 'mean'
            signal_dropout_prob (float): signal dropout probability, either
                frame dropout or spectral feature dropout
            signal_masking (bool):  whether to mask signal during inference

            How to use example:
            speech_enc = AcousticEncoder(
                spectral_dim=13,
                num_channels=[128, 256, 512],
                filter_sizes=[5, 10, 10],
                stride_steps=[1, 1, 1],
                pooling_type='max',

                # this will apply frame dropout with 0.2 prob
                signal_dropout_prob=0.2,
                dropout_frames=True,
                dropout_spectral_features=False,
                mask_signal= False
            ):
        """
        super(AcousticEncoder, self).__init__()
        self.spectral_dim = spectral_dim
        self.output_dim = output_dim
        self.max_num_frames = max_num_frames
        # self.signal_dropout_prob = signal_dropout_prob
        self.unit_dropout_prob = unit_dropout_prob
        self.pooling_type = pooling_type
        # self.dropout_frames = dropout_frames
        # self.dropout_spectral_features = dropout_spectral_features
        # self.mask_signal = mask_signal

        # signal dropout_layer
        # if self.dropout_frames: # if frame dropout is enableed
        #     self.signal_dropout = FrameDropout(self.signal_dropout_prob)

        # elif self.dropout_spectral_features: # if spectral dropout is enabled
        #     self.signal_dropout = SpectralDropout(self.signal_dropout_prob)

        # # frame reversal layer
        # self.frame_reverse = FrameReverse()

        # # frame reversal layer
        # self.frame_shuffle = FrameShuffle()

        # # add noise layer
        # self.noise = GaussianNoise()

        # Convolutional layer  1
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.spectral_dim,
                out_channels=num_channels[0],
                kernel_size=filter_sizes[0],
                stride=stride_steps[0]),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU(),
            nn.Dropout(self.unit_dropout_prob)
        )

        # Convolutional layer  2
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=filter_sizes[1],
                stride=stride_steps[1]),
            nn.BatchNorm1d(num_channels[1]),
            nn.ReLU(),
            nn.Dropout(self.unit_dropout_prob)
        )

        # Convolutional layer  3
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=filter_sizes[2],
                stride=stride_steps[2]),
            nn.BatchNorm1d(num_channels[2]),
            nn.ReLU(),
            nn.Dropout(self.unit_dropout_prob)
        )

        # determine the output dimensionality of the resulting tensor
        # this only works if all stride steps are set to 1
        shrinking_dims = sum([(i - 1) for i in filter_sizes[:]])
        out_dim = self.max_num_frames - shrinking_dims
        #print('out_dim', out_dim)

        if self.pooling_type == 'max':
            self.PoolLayer =  nn.Sequential(
                nn.MaxPool1d(kernel_size=out_dim, stride=1),
                #nn.Dropout(self.unit_dropout_prob)
            )


        elif self.pooling_type == 'avg':
            self.PoolLayer =  nn.Sequential(
                nn.AvgPool1d(kernel_size=out_dim, stride=1),
                #nn.Dropout(self.unit_dropout_prob)
            )
        else:
            #TODO: implement other statistical pooling approaches
            raise NotImplementedError

        #self.dropout = nn.Dropout(unit_dropout_prob)

        # feedforward fully-connected layers
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=num_channels[2], out_features=self.output_dim),
        #     nn.Tanh(),
        #     # nn.Dropout(self.unit_dropout_prob),
        #     # nn.Linear(in_features=self.output_dim,  out_features=self.output_dim),
        #     # nn.Tanh(),
        # )

        # self.fc2 = nn.Sequential(
        #     nn.Linear(in_features=self.output_dim,
        #         out_features=self.output_dim),
        #     nn.ReLU()
        # )


    def forward(self,
        x_in,
        frame_dropout=False,
        feature_dropout=False,
        return_all_vectors=False
    ):
        """The forward pass of the acoustic encoder

        Args:
            x_in (torch.Tensor): an input data tensor with the shape
                (batch_size, spectral_dim, max_num_frames)
            frame_dropout (bool): whether to mask out frames (inference)
            feature_dropout (bool): whether to mask out features (inference)
        Returns:
            the resulting tensor. tensor.shape should be (batch, )
        """

        # apply signal dropout on the input (if any)
        # signal dropout, disabled on evaluating unless explicitly asked for
        # if self.training:
        #     x_in = self.signal_dropout(x_in)

        # signal masking during inference (explicit)
        # if self.eval and self.mask_signal:
        #     x_in = self.signal_dropout(x_in)


        conv1_f = self.conv1(x_in)
        conv2_f = self.conv2(conv1_f)
        #print(conv2_f.shape)
        conv3_f = self.conv3(conv2_f)

        # print(f"conv1_f, {conv1_f.shape}",
        #     f"conv2_f, {conv2_f.shape}",
        #     f"conv3_f, {conv3_f.shape}",
        # )

        # max pooling
        conv_features = self.PoolLayer(conv3_f).squeeze(dim=2)

        #fc1_vec = self.fc1(conv_features)

        # final acoutic vector (output from the acoustic encoder)
        acoustic_embedding = conv_features #self.fc(conv_features) #  #conv_features  #self.fc(conv_features)


        if return_all_vectors:
            return conv_features, acoustic_embedding

        return acoustic_embedding


##### CLASS SymbolicEncoder: a symbolic encoder
class SymbolicEncoder(nn.Module):
    def __init__(self,
        embedding_dim=64,
        hidden_state_dim=512,
        output_dim=1024,
        n_layers=1,
        vocab_size=95,
        dropout_prob=0.2
    ):
        """
        Args:
            embedding_dim (int): dim of symbol embedding
            hidden_state_dim (int): dim of recurrent hidden state
            n_layers (int): number of layers in the recurrent model
            dropout_prob (float): dropout probability
        """
        super(SymbolicEncoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.symbol_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.recurrent_layer = nn.GRU(
            embedding_dim,
            hidden_state_dim,
            n_layers,
            dropout=0.0,
            bidirectional=True,
            batch_first=True
        )
        # self.dropout = nn.Dropout(dropout_prob)

        # fully connected layer
        # self.fc = nn.Sequential(
        #     nn.Dropout(dropout_prob),
        #     nn.Linear(in_features=2*hidden_state_dim, out_features=output_dim),
        #     nn.Tanh()
        # )


    def forward(self,
        x_in,
        initial_hidden_state
    ):
        """The forward pass of the encoder

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch_size, embedding_dim, sequence_len)
        Returns:
            the resulting tensor. tensor.shape should be (batch, sequence_len)
        """

        # permute input for recurrent computation
        # switch between embedding_dim <-> sequence_len

        #print(x_in.shape)
        #x_in = torch.LongTensor(x_in) #x_in = x_in.long()

        embeddings = self.symbol_embeddings(x_in)

        #embeddings = embeddings.permute(0, 2, 1)

        #print('embeddings', embeddings.shape)

        # recurrent_vectors shape: [batch_size x sequence_len x hidden_state_dim]
        recurrent_vectors, h_n = self.recurrent_layer(embeddings, initial_hidden_state)


        # last_recurrent shape: [batch_size x 1 x 2*hidden_state_dim]
        # this returns the concat of the last hidden states in
        # the forward and reverse directions of the recurrence
        last_recurrent = recurrent_vectors[:, -1, :]

        symbolic_embedding = last_recurrent #self.fc(last_recurrent)

        return symbolic_embedding

    def init_hidden(self, batch_size, device, bidirectional=True):
        """
        Given batch size & the device where the training of NN is taking place,
        return a proper (zero) initialiization for the LSTM model as (h_0, c_0).
        """

        # each layers requires its own initial states,
        # if bidirctional, multiply by 2
        N = self.n_layers * 2 if bidirectional else self.n_layers

        state = torch.zeros(N, batch_size, self.hidden_state_dim).to(device)
        return state



##### CLASS MultiViewEncoder:
class MultiViewEncoder(nn.Module):
    """A multi-view encoder based on acoustic & symbolic encoders."""
    def __init__(self, acoustic_encoder, symbolic_encoder):
        """
        Args:
            acoustic_encoder (SymbolicEncoder): module to get acoustic view
            symbolic_encoder (AcousticEncoder): module to get symbolic view
        """
        super(MultiViewEncoder, self).__init__()
        self.acoustic_encoder = acoustic_encoder
        self.symbolic_encoder = symbolic_encoder


    def forward(self, acoustic_in, symbolic_in, device):

        batch_size, _ = symbolic_in.shape

        _hidden_0 = self.symbolic_encoder.init_hidden(
            batch_size,
            device=device
        )

        symbolic_view = self.symbolic_encoder(
            #torch.LongTensor(symbolic_in), #.unsqueeze(0)
            symbolic_in,
            _hidden_0
        )

        _hidden_0 = self.acoustic_encoder.init_hidden(
            batch_size,
            device=device
        )

        acoustic_view = self.acoustic_encoder(acoustic_in, _hidden_0) 

        return acoustic_view, symbolic_view


##### CLASS CLSpeechEncoder:
class CLSpeechEncoder(nn.Module):
    """A single-view encoder based on ngram classification objective."""
    def __init__(self, acoustic_encoder, hidden_dim, num_classes):
        """
        Args:
            acoustic_encoder (AcousticEncoder): module to get acoustic view
        """
        super(CLSpeechEncoder, self).__init__()
        self.acoustic_encoder = acoustic_encoder

        self.cls_layers = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes)
            #nn.Dropout(dropout_prob),
            #nn.Linear(in_features=output_dim, out_features=output_dim),
            #nn.ReLU()
        )

        #self.cls_layer = nn.Linear(hidden_dim, num_classes)


    def forward(self, acoustic_in, encoder_type, device=None, return_vector=False):

        if encoder_type in ['LSTM', 'BiGRU', 'BiLSTM']:
            # if a recurrent encoder, initial hidden state is required
            #print(acoustic_in.shape)
            batch_size, _,_ = acoustic_in.shape

            _hidden_0 = self.acoustic_encoder.init_hidden(
                batch_size,
                device=device
            )

            acoustic_emb = self.acoustic_encoder(
                acoustic_in,
                _hidden_0
            )
        else:
            acoustic_emb = self.acoustic_encoder(acoustic_in)

        logits = self.cls_layers(acoustic_emb)

        # pass through classification layer
        if return_vector:
            return logits, acoustic_emb

        return logits



##### CLASS WordPairEncoder:
class WordPairEncoder(nn.Module):
    """A single-view encoder based on pair contrastive objective."""
    def __init__(self, acoustic_encoder, encoder_type):
        """
        Args:
            acoustic_encoder (AcousticEncoder): module to get acoustic view
        """
        super(WordPairEncoder, self).__init__()
        self.acoustic_encoder = acoustic_encoder
        self.encoder_type = encoder_type


    def forward(self, acoustic_ank, acoustic_pos, device=None):

        if self.encoder_type in ['LSTM', 'BiGRU', 'BiLSTM']:
            # if a recurrent encoder, initial hidden state is required
            #print(acoustic_ank.shape)
            batch_size, _,_ = acoustic_ank.shape

            _hidden_0 = self.acoustic_encoder.init_hidden(
                batch_size,
                device=device
            )

            acoustic_emb_ank = self.acoustic_encoder(acoustic_ank, _hidden_0)
            acoustic_emb_pos = self.acoustic_encoder(acoustic_pos, _hidden_0)
        else:
            acoustic_emb_ank = self.acoustic_encoder(acoustic_ank)
            acoustic_emb_pos = self.acoustic_encoder(acoustic_pos)


        return acoustic_emb_ank, acoustic_emb_pos


class AcousticEncoderLSTM(nn.Module):
    def __init__(self,
        spectral_dim=13,
        max_num_frames= 384,
        hidden_state_dim=256,
        output_dim=512,
        n_layers=4,
        unit_dropout_prob=0.2,
        dropout_frames=False,
        dropout_spectral_features=False,
        mask_signal=False,
        dropout_prob=0.2,
        signal_dropout_prob=0.0
    ):
        """
        Args:
            feature_dim (int): size of the feature vector
            num_classes (int): num of classes or size the softmax layer
            )
        """
        super(AcousticEncoderLSTM, self).__init__()
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_state_dim = hidden_state_dim


        self.recurrent_layer = nn.LSTM(
            input_size=spectral_dim,
            hidden_size=hidden_state_dim,
            num_layers=n_layers,
            dropout=dropout_prob,
            bidirectional=False,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout_prob)

        # fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(in_features=hidden_state_dim, out_features=output_dim),
            nn.Tanh()
            #nn.Dropout(dropout_prob)
        )


    def forward(self, x_in, initial_hidden_state):
        """The forward pass of the encoder

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch_size, embedding_dim, sequence_len)
        Returns:
            the resulting tensor. tensor.shape should be (batch, sequence_len)
        """

        x_in = x_in.float()

        # input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
		# input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)

        # print(x_in.shape) >> torch.Size([256, 39, 128]) (batch, feature_dim, seq_len)

        # permute input for recurrent computation
        x_in = x_in.permute(0, 2, 1)


        # print(x_in.shape) >> torch.Size([128, 256, 39])
        
        # LSTM Inputs: input, (h_0, c_0)
        # input of shape (seq_len, batch, input_size)
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)

        # LSTM Outputs: output, (h_n, c_n)
        # output of shape (seq_len, batch, num_directions * hidden_size)
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        (h_0, c_0) = initial_hidden_state
        lstm_out, (h_n, c_n) = self.recurrent_layer(x_in, (h_0, c_0))
       

        #lstm_vec =  self.dropout(lstm_out[:, -1, :]) #
        lstm_vec = lstm_out[:, -1, :]
        #print(lstm_vec.shape)  >> torch.Size([256, 1024])

        #print('last_hidden', last_hidden.shape)

        #z = self.fc_layers(lstm_out)
        #print('z', z.shape)

        acoustic_embedding = self.fc(lstm_vec)

        return lstm_vec
    

    def init_hidden(self, batch_size, device, bidirectional=False):
        """
        Given batch size & the device where the training of NN is taking place,
        return a proper (zero) initialization for the LSTM model as (h_0, c_0).
        """

        # each layers requires its own initial states,
        # if bidirctional, multiply by 2
        N = self.n_layers if bidirectional else self.n_layers

        state = (
            torch.zeros(N, batch_size, self.hidden_state_dim).to(device),
            torch.zeros(N, batch_size, self.hidden_state_dim).to(device),
        )

        return state


class AcousticEncoderBiLSTM(nn.Module):
    def __init__(self,
        spectral_dim=13,
        max_num_frames= 384,
        hidden_state_dim=256,
        output_dim=512,
        n_layers=4,
        unit_dropout_prob=0.2,
        dropout_frames=False,
        dropout_spectral_features=False,
        mask_signal=False,
        signal_dropout_prob=0.0
    ):
        """
        Args:
            feature_dim (int): size of the feature vector
            num_classes (int): num of classes or size the softmax layer
            )
        """
        super(AcousticEncoderBiLSTM, self).__init__()
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_state_dim = hidden_state_dim


        self.recurrent_layer = nn.LSTM(
            input_size=spectral_dim,
            hidden_size=hidden_state_dim,
            num_layers=n_layers,
            dropout=unit_dropout_prob,
            bidirectional=True,
            batch_first=True
        )
        
        #self.dropout = nn.Dropout(dropout_prob)

        # fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(in_features=2*hidden_state_dim, out_features=output_dim),
            nn.Tanh()
            #nn.Dropout(dropout_prob),
            #nn.Linear(in_features=output_dim, out_features=output_dim),
            #nn.Tanh()
        )


    def forward(self, x_in, initial_hidden_state):
        """The forward pass of the encoder

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch_size, embedding_dim, sequence_len)
        Returns:
            the resulting tensor. tensor.shape should be (batch, sequence_len)
        """

        x_in = x_in.float()

        # input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
		# input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)

        # print(x_in.shape) >> torch.Size([256, 39, 128]) (batch, feature_dim, seq_len)

        # permute input for recurrent computation
        x_in = x_in.permute(0, 2, 1)


        # print(x_in.shape) >> torch.Size([128, 256, 39])
        
        # LSTM Inputs: input, (h_0, c_0)
        # input of shape (seq_len, batch, input_size)
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)

        # LSTM Outputs: output, (h_n, c_n)
        # output of shape (seq_len, batch, num_directions * hidden_size)
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        (h_0, c_0) = initial_hidden_state
        lstm_out, (h_n, c_n) = self.recurrent_layer(x_in, (h_0, c_0))
       

        lstm_vec =  lstm_out[:, -1, :] # self.dropout(lstm_out[:, -1, :]) #
        #lstm_vec = lstm_out[:, -1, :]
        #print(lstm_vec.shape)  >> torch.Size([256, 1024])

        #print('last_hidden', last_hidden.shape)

        #z = self.fc_layers(lstm_out)
        #print('z', z.shape)

        #acoustic_embedding = self.fc(lstm_vec)

        return lstm_vec #acoustic_embedding
    

    def init_hidden(self, batch_size, device, bidirectional=True):
        """
        Given batch size & the device where the training of NN is taking place,
        return a proper (zero) initialization for the LSTM model as (h_0, c_0).
        """

        # each layers requires its own initial states,
        # if bidirctional, multiply by 2
        N = 2*self.n_layers if bidirectional else self.n_layers

        state = (
            torch.zeros(N, batch_size, self.hidden_state_dim).to(device),
            torch.zeros(N, batch_size, self.hidden_state_dim).to(device),
        )

        return state


class AcousticEncoderBiGRU(nn.Module):
    def __init__(self,
        spectral_dim=13,
        max_num_frames= 384,
        hidden_state_dim=256,
        output_dim=512,
        n_layers=4,
        unit_dropout_prob=0.2,
        dropout_frames=False,
        dropout_spectral_features=False,
        mask_signal=False,
        signal_dropout_prob=0.0
    ):
        """
        Args:
            feature_dim (int): size of the feature vector
            num_classes (int): num of classes or size the softmax layer
            )
        """
        super(AcousticEncoderBiGRU, self).__init__()
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_state_dim = hidden_state_dim
        self.unit_dropout_prob=unit_dropout_prob


        self.recurrent_layer = nn.GRU(
            input_size=spectral_dim,
            hidden_size=hidden_state_dim,
            num_layers=n_layers,
            dropout=unit_dropout_prob,
            bidirectional=True,
            batch_first=True
        )
        
        #self.dropout = nn.Dropout(unit_dropout_prob)

        #fully connected layer
        # self.fc = nn.Sequential(
        #     #nn.Dropout(unit_dropout_prob),
        #     nn.Linear(in_features=2*hidden_state_dim, out_features=output_dim),
        #     nn.Tanh(),
            #nn.Dropout(unit_dropout_prob),
            # nn.Linear(in_features=output_dim, out_features=output_dim),
            # nn.ReLU()
        #)


    def forward(self, x_in, initial_hidden_state):
        """The forward pass of the encoder

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch_size, embedding_dim, sequence_len)
        Returns:
            the resulting tensor. tensor.shape should be (batch, sequence_len)
        """

        x_in = x_in.float()

        # input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
		# input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)

        # print(x_in.shape) >> torch.Size([256, 39, 128]) (batch, feature_dim, seq_len)

        # permute input for recurrent computation
        x_in = x_in.permute(0, 2, 1)


        # print(x_in.shape) >> torch.Size([128, 256, 39])
        
        # GRU Inputs: input, h_0
        # input of shape (seq_len, batch, input_size)
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        
        # GRU Outputs: output, h_n
        # output of shape (seq_len, batch, num_directions * hidden_size)
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        h_0 = initial_hidden_state
        rec_out, h_n = self.recurrent_layer(x_in, h_0)
       
        rec_vec =  rec_out[:, -1, :] #self.dropout(rec_out[:, -1, :]) #
        #lstm_vec = lstm_out[:, -1, :]
        #print(lstm_vec.shape)  >> torch.Size([256, 1024])

        #print('last_hidden', last_hidden.shape)

        #z = self.fc_layers(lstm_out)
        #print('z', z.shape)

        acoustic_embedding = rec_vec

        return acoustic_embedding
    

    def init_hidden(self, batch_size, device, bidirectional=True):
        """
        Given batch size & the device where the training of NN is taking place,
        return a proper (zero) initialization for the LSTM model as (h_0, c_0).
        """

        # each layers requires its own initial states,
        # if bidirctional, multiply by 2
        N = 2*self.n_layers if bidirectional else self.n_layers

        state = (
            torch.zeros(N, batch_size, self.hidden_state_dim).to(device)
        )

        return state


##### CLASS PhoneticDecoder:
class PhoneticDecoder(nn.Module):
    def __init__(self,
        embedding_dim=64,
        hidden_state_dim=512,
        n_layers=1,
        vocab_size=95,
        dropout_prob=0.2
    ):
        """
        Args:
            embedding_dim (int): dim of phoneme embedding
            hidden_state_dim (int): dim of recurrent hidden state
            n_layers (int): number of layers in the recurrent model
        """
        super(PhoneticDecoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.phoneme_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.recurrent_layer = nn.GRUCell(
            input_size=embedding_dim,
            hidden_size=hidden_state_dim
        )

        self.fc_out = nn.Linear(hidden_state_dim, vocab_size)
        
        #self.dropout = nn.Dropout(dropout)


    def forward(self, x_in, previous_hidden):

        # permute
        #print('x_in.shape', x_in.shape) 

        #x_in = x_in.permute(1, 0)
        
        embedding = self.phoneme_embeddings(x_in) # self.dropout()
                
        hidden_state = self.recurrent_layer(embedding, previous_hidden)
        
        prediction = self.fc_out(hidden_state)
        
        return prediction, hidden_state


##### CLASS Seq2SeqEncoder
class Seq2SeqEncoder(nn.Module):
    def __init__(self, acoustic_encoder, phonetic_decoder, encoder_type='BiGRU', device='cpu'):
        super(Seq2SeqEncoder, self).__init__()
        
        self.acoustic_encoder = acoustic_encoder
        self.phonetic_decoder = phonetic_decoder
        self.encoder_type=encoder_type
        self.device = device

        output_dim = acoustic_encoder.output_dim

        self.fc = nn.Sequential(
            nn.Dropout(acoustic_encoder.unit_dropout_prob),
            nn.Linear(in_features=output_dim, out_features=output_dim),
            nn.Tanh(),
            # nn.Dropout(acoustic_encoder.unit_dropout_prob),
            # nn.Linear(in_features=output_dim, out_features=output_dim),
            # nn.Tanh()
        )

        self.teacher_force=True
        self.teacher_forcing_ratio = 1.0
        
        #assert 2*acoustic_encoder.hidden_state_dim == phonetic_decoder.hidden_state_dim, \
        #    "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, acoustic_in, y_sequence_out, inference=False):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        #print('y_sequence_out', y_sequence_out.shape)
        
        batch_size = y_sequence_out.shape[0]

        if self.encoder_type in ['LSTM', 'BiGRU', 'BiLSTM']:

            _hidden_0 = self.acoustic_encoder.init_hidden(
                batch_size,
                device=self.device
            )

            acoustic_emb = self.fc(self.acoustic_encoder(acoustic_in, _hidden_0))

        else:
            acoustic_emb = self.fc(self.acoustic_encoder(acoustic_in)) #self.acoustic_encoder(acoustic_in) #

        if inference:
            return acoustic_emb

        # initialize decoder hidden state with acoustic vector 
        decoder_hidden_state = acoustic_emb

        

        y_sequence_len = y_sequence_out.shape[1]


        output_vocab_size = self.phonetic_decoder.vocab_size
        
        #tensor to store decoder outputs
        outputs = torch.zeros(y_sequence_len, batch_size, output_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        #hidden_state = self.acoustic_encoder(acoustic_in)
        
        #first input to the decoder is the <BEGIN> token
        recurrent_input = y_sequence_out[:,0]

        #print(recurrent_input)
        
        for t in range(1, y_sequence_len):
            
            #insert input token embedding, previous hidden 
            #receive output tensor (predictions) and new hidden 
            output, decoder_hidden_state = self.phonetic_decoder(recurrent_input, decoder_hidden_state)


            #print('output, decoder_hidden_state', output.shape, decoder_hidden_state.shape)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            #teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            #top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            recurrent_input = y_sequence_out[:,t] #if self.teacher_force else top1
        
        return outputs, acoustic_emb