# Helper functions to train neural models

# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

import os
from collections import defaultdict, Counter
import sys
import torch
import torch.nn.functional as F
import numpy as np
import faiss

from scipy.spatial import distance
import scipy.stats as stats


def make_train_state(args):
    return {
        'stop_early': False,
        'early_stopping_step': 0,
        'early_stopping_best_val': 1e8,
        'learning_rate': args['training_hyperparams']['learning_rate'],
        'epoch_index': 0,
        'train_loss': [],
        'val_loss': [],
        'val_acoustic_mAP': [], 
        'val_crossview_mAP': [],
        'model_filename': args['model_state_file']
    }


def update_train_state(args, model, train_state):
    """Handle the training state updates.
    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better
    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """
    # save model
    torch.save(model.state_dict(), train_state['model_filename'] + '_' + \
        str(train_state['epoch_index'] + 1) + '.pth')

    # save model after first epoch
    if train_state['epoch_index'] == 0:
        train_state['stop_early'] = False
        train_state['best_val_acoustic_mAP'] = train_state['val_acoustic_mAP'][-1]

    # after first epoch check early stopping criteria
    elif train_state['epoch_index'] >= 1:
        score_mAP = train_state['val_acoustic_mAP'][-1]

        # if acc decreased, add one to early stopping criteria
        if score_mAP <= train_state['best_val_acoustic_mAP']:
            # Update step
            train_state['early_stopping_step'] += 1

        else: # if acc improved
            train_state['best_val_acoustic_mAP'] = train_state['val_acoustic_mAP'][-1]

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        early_stop = train_state['early_stopping_step'] >= \
            args['training_hyperparams']['early_stopping_criteria']

        train_state['stop_early'] = early_stop

    return train_state


def compute_accuracy(y_pred, y_target):
    #y_target = y_target.cpu()
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def get_predictions_and_trues(y_pred, y_target):
	"""Return indecies of predictions. """
	_, y_pred_indices = y_pred.max(dim=1)

	pred_labels = y_pred_indices.tolist()
	true_labels = y_target.tolist()

	return (pred_labels, true_labels)


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)




def get_hard_negatives(index_mat, query_mat, labels, negative_strategy='hard', debug=False):
    """Get N, M, and L, return hard negatives"""

    index_mat = index_mat.cpu().detach().numpy()
    query_mat =  query_mat.cpu().detach().numpy()

    faiss.normalize_L2(index_mat)
    
    faiss.normalize_L2(query_mat)

    d = index_mat.shape[1] # this should be the size of the embedding

    # build index
    faiss_idx = faiss.IndexFlatIP(d)
    faiss_idx.add(index_mat)

    k = index_mat.shape[0] # we want to see k nearest neighbors
    D, I = faiss_idx.search(query_mat, k)

    #print(I)
    #print(D)

    # get hard negative for each anchor point
    # with one constraint; the negative should not
    # correspond to the same label

    idx2label = {i:l for i, l in enumerate(labels)}

    #label2idx = {'_'.join([str(i),str(l)]):i for i, l in idx2label.items()}

    #print(label2idx)


    labeling_func = lambda i: idx2label[i]

    vfunc = np.vectorize(labeling_func)

    I2 = vfunc(I)

    # [f(x) if condition else g(x) for x in sequence]
    filtering_func = lambda i, j_list: [1 if i!=j else 0 for j in j_list] # else 0

    neg_list = []

    for row, label in zip(I2, labels):
        neg_list.append(filtering_func(label, row))

    #print(neg_list)

    sorted_neg_by_index = []

    for row1, row2, neg in zip(I, I2, neg_list):

        batch_items = []

        for w_idx, w_lbl, isNeg in zip(row1, row2, neg):
            #print((w_idx, w_lbl, isSame), end=' ')

            if isNeg:
                batch_items.append(w_idx)

        sorted_neg_by_index.append(batch_items)
        #print()


    #for item in sorted_neg_by_index:
    #print(item)

    sorted_neg_by_label = [list(map(labeling_func, l)) for l in sorted_neg_by_index]

    #for item in zip(sorted_neg_by_label, sorted_neg_by_index):
    #print(item)

    if debug:
        for i, (negative_labels, negative_indicies) in enumerate(zip(sorted_neg_by_label, sorted_neg_by_index)):
            print(f"{i:<7}{idx2label[i]:<7} {label2word[idx2label[i]]:<10}"
                f"{' '.join(str(m) for m in negative_labels):<35}"
                f"{' '.join(str(m) for m in negative_indicies):<35}")
            #f"\t {' '.join(str(label2word[m]) for m in ll):<50}")

        #print(f"{i:<5} {word2label[w]:<5} {w:<10}")



    if negative_strategy == 'hard':
        negative_indicies = [n[0] for n in sorted_neg_by_index]
        return negative_indicies

    if negative_strategy == 'random':
        negative_indicies = []

        for row in sorted_neg_by_index:
            rand_idx = np.random.randint(len(row))

            negative_indicies.append(row[rand_idx])

        return negative_indicies



def average_precision(database_mat, query_mat, database_labels, query_labels, index2word, single_view=False):
    """Get two matrices and list of labels, return AP"""

    # L2 normalize to allow for cosine distance measure
    faiss.normalize_L2(database_mat)
    faiss.normalize_L2(query_mat)

    # d is the dim of the embedding
    d = database_mat.shape[1]

    # build index wuing FAISS
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(database_mat)
    #print(faiss_index.ntotal)

    # query the index
    k = database_mat.shape[0] # we want to see k nearest neighbors

    if single_view:
        D, ranked_candidates_by_index = faiss_index.search(query_mat, k) #[1:, :]

    else:
        D, ranked_candidates_by_index = faiss_index.search(query_mat, k)

    if single_view: 
        print(ranked_candidates_by_index[0][:10])
        #     print(index2word[idx], end='\t')
        # print()

        for d in D[0][:10]:
            print(f"{d:.9f}", end='\t') 
        print()

        #print(len(database_labels))
        #print(database_mat[0])


    # make a dict for DB index -->  word label
    faiss_index_to_word_label = {i:l for i, l in enumerate(database_labels)}

    # make a function to obtain word label of the index, then vectorize
    get_word_label = lambda i: faiss_index_to_word_label[i]
    get_word_label = np.vectorize(get_word_label)


    ranked_candidates_by_word_label = get_word_label(ranked_candidates_by_index)
    if single_view: 
        #print(index2word[database_labels[0]].encode('utf-8'))

        # for idx in ranked_candidates_by_word_label[0][:10]:
        #     print(index2word[idx].encode('utf-8'), end='\t')

        # print()

        # print(index2word[database_labels[1]].encode('utf-8'))

        # for idx in ranked_candidates_by_word_label[1][:10]:
        #     print(index2word[idx].encode('utf-8'), end='\t')
        # print()

        # print(index2word[database_labels[2]].encode('utf-8')) #.decode(sys.stdout.encoding)

        # for idx in ranked_candidates_by_word_label[2][:10]:
        #     print(index2word[idx].encode('utf-8'), end='\t')
        # print()

        # print(index2word[database_labels[3]].encode('utf-8'))

        # for idx in ranked_candidates_by_word_label[3][:10]:
        #     print(index2word[idx].encode('utf-8'), end='\t')
        # print()

        for i in range(20):
            print(index2word[database_labels[i]].encode('utf-8')) #.decode('utf-8')

            for idx in ranked_candidates_by_word_label[i][:10]:
                print(index2word[idx].encode('utf-8'), end='\t') #.decode('utf-8')
            print()



    AP_values = []



    # loop over each query word and the ranked candidates
    for query_word, candidates in zip(query_labels, ranked_candidates_by_word_label):
            # chech if single-view so the word vector of
            # the same sample is not included in the candidate list
            if single_view:

                # when sim is all 0, this does not hold! 
                candidates = candidates[1:]

                # if the query word has matching sample then go to next sample
                # this skips words that are not a part of a pair
                if query_word not in candidates:
                    continue

                #print('length:', len(candidates), 'Query:', query_word,  ' '.join(str(i) for i in candidates))
            
            #from random import shuffle
            #shuffle(candidates)

            # make a vector of the word label repeated as the candidate list
            target_word_label = np.array(
                [query_word for i in range(len(candidates))]
            )

            # make a binary [0 1 .. ] array of whether word is a match
            matches = 1*np.equal(target_word_label, candidates)

            # Calculate precision for this query word
            precision = np.cumsum(matches)/np.arange(1, len(matches) + 1)

            AP = np.sum(precision * matches) / sum(matches)

            AP_values.append(AP)    
    
    return AP_values



def compute_kendalls_tau(database_mat, query_mat, word2dist, index2word):
    
    # L2 normalize to allow for cosine distance measure
    faiss.normalize_L2(database_mat)
    faiss.normalize_L2(query_mat)

    # d is the dim of the embedding
    d = database_mat.shape[1]

    # build index wuing FAISS
    print('Build FAISS index ...')
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(database_mat)
    #print(faiss_index.ntotal)

    # query the index
    k = database_mat.shape[0] # we want to see k nearest neighbors
    
    
    # perform search 
    print('Query FAISS index ...')
    D, I = faiss_index.search(query_mat, k)
    
    
    # compute rank by phonological distance 
    print('Compute rank by PWLD similarity ...')
    PWLD_rank2word = defaultdict(lambda: defaultdict())
    
    for w1 in word2dist.keys():

        for i, (w2, d) in enumerate(dict(sorted(word2dist[w1].items(), key=lambda item: item[1])).items()):

            PWLD_rank2word[w1][i] = w2
            
    
    # switch key vs value in PWLD_rank2word
    word2PWLD_rank = defaultdict(lambda: defaultdict())

    for w1 in word2dist.keys():
        word2PWLD_rank[w1] = {v:k for k, v in PWLD_rank2word[w1].items()}
        
        
    # obtain rank by cosine similarity
    cos_rank2word = defaultdict(lambda: defaultdict())

    for q_idx in range(len(I)):

        for i, hit_idx in enumerate(I[q_idx]):
            cos_rank2word[q_idx][i] = index2word[hit_idx]

    
    # obtain kendall's tau for each word query
    tau_values = []
    spr_values = []
    word2tau = {}

    for dict_idx in cos_rank2word:

        phon_ranks = []
        cosd_ranks = []

        for rank, word in list(cos_rank2word[dict_idx].items())[1:]:

            phon_ranks.append(word2PWLD_rank[index2word[dict_idx]][word])
            cosd_ranks.append(rank)

        tau, p_value = stats.kendalltau(cosd_ranks, phon_ranks)
        spr, p_value = stats.spearmanr(cosd_ranks, phon_ranks)

        word2tau[dict_idx] = tau

        tau_values.append(tau)
        spr_values.append(spr)
    
    
    return tau_values, spr_values



def triplet_loss_func(a, p, n, margin=0.4):
    """
    Givne three tensors, return triplet margin loss.
    """

    # get batch size, if batch size == 1, return 0.0 loss
    m = a.shape[0]

    if m == 1:
        return torch.tensor(0.0, requires_grad=True).cuda()

    distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)

    loss = F.relu(margin + distance_function(a, p) - distance_function(a, n))

    torch.set_printoptions(precision=14)
    return loss.sum()



# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)



# takes in a module and applies the specified weight initialization
def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(-0.05, 0.05)
        m.bias.data.fill_(0)
