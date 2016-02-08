"""
Functions that wlap KALDI functions.

Author: Naohiro Tawara
Contact: tawara@ttic.edu
Date: 2016
"""

import logging
import numpy as np
import sys
from kaldi.io import KaldiScp, KaldiArk, KaldiNnet
from kaldi.commands import KaldiCommand


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch=logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

#-----------------------------------------------------------------------------#
#                         TIMIT  DATASET I/O FUNCTIONS                        #
#-----------------------------------------------------------------------------#

def splice(X, offsets):
    XX=[]
    N,D =X.shape
    L=len(offsets)
    for t in xrange(N):
        XX.append(np.reshape(X[map(lambda x: min(max(0,t+x), N-1), offsets)], (1, L*D)))
    return np.vstack(XX)

def __get_word_to_i__(labels):
    i_type = 0
    word_to_i_map = {}
    for label in labels:
        if not word_to_i_map.has_key(label):
            word_to_i_map[label] = i_type
            i_type += 1
    return word_to_i_map

def load_labeled_data(ark, pdf, offsets=[0]):
    X = []
    labels = []
    ali = {key:vec for key, vec in pdf}
    for key, data in ark:
        labels.append(ali[key])
        for x in splice(data, offsets):
            X.append(x)
    return np.vstack(X), np.hstack(labels)

def load_data(ark, offsets=[0]):
    X = []
    Nf={}
    s_index =0
    for key, data in ark:
        for x in splice(data, offsets):
            X.append(x)
        Nf[key] = (s_index,s_index+data.shape[0]-1)
        s_index +=data.shape[0]
    return np.vstack(X), Nf

def load_timit_labelled_kaldi(ark, pdf, offsets=[0]):
    """
    Load the TIMIT frames with their labels from .scp file.
    Each frame is concatinated with its neighbor frames
    (e.x. offsets=[-2,-1,0,1,2] then each frame is concatenated with +-2 frames 
     and generating dim*5 dimensional vectors).
    If ratio is less than 1, some labels will be removed for semi-supervised experiment.
    The ratio determines the ratio of labeled data to whole data (unlabeled + labeled).
    In this time, the minimum number of samples in each class will be min_count.
    If foldings_file is given, some phones are folded into specific phone.

    Return: feature matrix, 
    """
    train_x, train_y = load_labeled_data(ark, pdf, offsets)
#    if word_to_i is None:
#        word_to_i = __get_word_to_i__(train_labels)

    # Convert labels to word IDs
#    train_y = np.asarray([word_to_i[label] for label in train_labels],dtype=np.int32)

    return (train_x, train_y)

def load_nnet(filename):
    return [layer for layer in KaldiNnet(filename)]

'''
class KaldiDataset:
    def __init__(self, filename, in_memory):
        if filename.splitext(filename)[-1] == "scp":
            self.kaldi_obj = KaldiScp(filename)
        elif filename.splitext(filename)[-1] == "ark":
            self.kaldi_obj = KaldiArk(filename)
        self.in_memory=in_memory
 
       if self.in_memory:
            self.data = [line for line in self.kaldi_obj]
        else:
            self.data = []

    def get_data(self)

class KaldiIterableDataset(KaldiDataset):
    def __init__(self, filename, in_memory=False):
        super(KaldiIterableDataset, self).__init__(filename, in_memory)

class KaldiIndexes:

class Kaldipairwise:
''' 

#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def convert_nnet_to_conf(filename):
    layers_specs =[]
    layers = load_nnet(filename)
    i =0
    while i <len(layers):
        if layers[i]['func'] == 'AffineTransform':
            d={}
            d['dimensions'] =[layers[i]['dim'][1],layers[i]['dim'][0]]
            d['type'] = 'full'
            d['activation'] = layers[i+1]['func']
            d['W'] = layers[i]['W']
            d['b'] = layers[i]['b']
            layers_specs.append(d)
            i=i+1
        i=i+1
    return layers_specs

def get_mss_set(train_x, train_y, ratio=1.0, min_counts=1):
    # Remove labels from some samples for semi-supervised setting
    # The number of non-labeled samples is determined to be equal among clusters.
    # Return: Indexes of labeled data
    ndata=train_x.shape[0]
    if ratio == 1.0:
        logger.info("Using all data: %d" % ndata)
        i_labeled=np.array(range(0,ndata))
    else:
        n_classes = train_y.max() + 1
        min_samples=sys.maxint
        max_samples=0
        avg_samples=0
        indices=np.array(range(0,ndata))
        i_labeled = []
        n_labeled_all = 0
        for c in range(n_classes):
            n_samples= round(sum(train_y==c) * ratio) \
                if round(sum(train_y==c) * ratio) >= min_count else min_count
            i = (indices[train_y == c])[:n_samples]
            i_labeled += list(i)
            avg_samples += n_samples
            min_samples = n_samples if n_samples < min_samples else min_samples
            max_samples = n_samples if n_samples > max_samples else max_samples
            n_labeled_all += n_samples
        logger.info('Minimum number of labeled samples: %d' % min_samples)
        logger.info('Maximum number of labeled samples: %d' % max_samples)
        logger.info('Average number of labeled samples: %d' % int(avg_samples / n_classes))
        logger.info('Total   number of labeled samples: %d' % n_labeled_all)
    return i_labeled

#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

if __name__ == "__main__":
#    logging.basicConfig(level=logging.DEBUG)
#    ali_to_pdf=KaldiCommand('bin/ali-to-pdf', option='timit/exp/tri3_ali/final.mdl')
#    pdf = ali_to_pdf << '\"gunzip -c timit/exp/tri3_ali/ali.*.gz |\"'
#    filename= '/data2/tawara/work/ttic/MyPython/src/kaldi/timit/data/fbank/train_tr90/feats.scp'
#    ark = KaldiScp(filename)
#    train, word_to_i = load_timit_labelled_kaldi(ark, pdf, [0])
#    ark.reset()
#    train, nf = load_data(ark,[0])

    filename ='timit/exp/fbank/dnn4_nn5_1024_cmvn_splice10_pretrain-dnn/final.nnet'
    layers = load_nnet(filename)
    print layers

