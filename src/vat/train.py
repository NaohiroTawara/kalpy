# coding: utf-8

# In[1]:
import sys
sys.path.append('../../')

from chainer import Variable, optimizers,cuda
import cupy


import logging
from kaldi.data import load_timit_labelled_kaldi, load_data
import time
import numpy
from chainer_tools.nn_parallel import NN_parallel

import pickle
import argparse
from argparse import ArgumentParser, Action
from datetime import datetime
import timeit
import numpy as np
import chainer.functions as F

from kaldi.commands import KaldiCommand
from kaldi.io import KaldiScp,KaldiArk

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
ch=logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

context_length=10
act='sigmoid'
logger = logging.getLogger(__name__)
default_options_dict = {
    "context_length": context_length,
    "data_dir": "data/icassp15.0",
    "njobs": 1,
    "rnd_seed": 42,
    "batch_size": 474,
    "n_max_epochs": 20,
    "model_dir": "models",
    "online_evaluation": False,
#    "layers_specs": [
#        {"type": "full", "dimensions": (23*(context_length*2+1), 1024), "activation": 'linear',
#         'BN': (True), 
#         'dropout': 0.5
#        },
#        {"type": "full", "dimensions": (1024, 1024), "activation": 'Sigmoid', 
#         'BN': (True), 
#         'dropout': 0.5
#        },
#        {"type": "full", "dimensions": (1024, 1024), "activation": 'Sigmoid', 
#         'BN': (True), 
#         'dropout': 0.5
#        },
#        {"type": "full", "dimensions": (1024, 1024), "activation": 'Sigmoid',
#         'BN': (True), 
#         'dropout': 0.5
#        },
#        {"type": "full", "dimensions": (1024, 1024), "activation": 'Sigmoid', 
#         'BN': (True), 
#         'dropout': 0.5
#        },
#        {"type": "full", "dimensions": (1024, 2032), "activation": 'linear', 
#         'BN': (True), 
#         'dropout': 0.5
#        },
#    ],
    "optimizer":optimizers.SGD(0.008),
#    "optimizer":optimizers.Adam(),

}
option_dict = default_options_dict


def calc_diff(X, post, num_rows):
    mat = X
    for i in xrange(num_rows):
        mat.data[i,post[i]] -= 1
    return mat

class Stats:
    def __init__(self):
        self.reset()
    def update(self, f):
        self.reset()
        min_value = f.min()
        max_value = f.max()
        sum_value = f.sum()
        if self.min_value > min_value:
            self.min_value = min_value
        if self.max_value < max_value:
            self.max_value = max_value
        self.sum_value = sum_value
    def str(self, dim):
        return ' min '+str(self.min_value) + \
            ', max '+str(self.max_value) + \
            ', mean '+str(float(self.sum_value) / float(dim))
    def reset(self):
        self.sum_value = 0
        self.min_value = sys.maxint
        self.max_value = -sys.maxint

def apply_batch(model, x_train, y_train, indexes, batchsize, train = True):
    sum_nll = 0
    sum_accuracy = 0
    test = not train
    out_stats   = Stats()
    in_stats    = Stats()
    gradW_stats = Stats()
    gradb_stats = Stats()
    diff_stats  = Stats()
    n_frm=0
    for i in xrange(0, len(indexes), batchsize):
        x_batch = x_train[indexes[i : i + batchsize]]
        y_batch = y_train[indexes[i : i + batchsize]]
#        output  = model.forward(x_batch, test=True)
        if train:
#            diff = calc_diff(F.softmax(output), y_batch, y_batch.shape[0]).data
            loss = model.loss_softmax(x_batch, y_batch, test=test)
            sum_nll  += cuda.to_cpu(loss.data) * batchsize
#            loss = nll # Maximum Likelihood Estimation
            model.zero_grads()
            loss.backward()
            model.update()

#            out_stats.update(output.data)
#            in_stats.update(x_batch)
#            diff_stats.update(diff)
#            gradW_stats.update(model.layers[0].ln.W.grad)
#            gradb_stats.update(model.layers[0].ln.b.grad)
#            n_frm += x_batch.shape[0]
#            if i ==0 or  (i / 25000) is not ((i+x_batch.shape[0])/25000):
#                print ("#### After "+ str(i) + " frames,")
#                print '  Input: ' + in_stats.str(n_frm*x_batch.shape[1])
#                print '  Output:' + out_stats.str(n_frm*x_batch.shape[1])
#                print '  Diff:  ' + diff_stats.str(n_frm*x_batch.shape[1])
#                print '  GradW: ' + gradW_stats.str(n_frm*x_batch.shape[1])
#                print '  Gradb: ' + gradb_stats.str(n_frm*x_batch.shape[1])
#                print'  ProgressLoss:' +str(loss.data) # print every 25k frames 

        sum_accuracy  += F.accuracy(model.forward(x_batch, test=True), Variable(cuda.to_gpu(y_batch,device=model.device_id[0]))).data * batchsize

#    if train:
#        print ("#### After "+ str(i) + " frames,")
#        print '  Input: ' + in_stats.str(n_frm*x_train.shape[1])
#        print '  Output:' + out_stats.str(n_frm*x_train.shape[1])
#        print '  Diff:  ' + diff_stats.str(n_frm*x_train.shape[1])
#        print '  GradW: ' + gradW_stats.str(n_frm*x_train.shape[1])
#        print '  Gradb: ' + gradb_stats.str(n_frm*x_train.shape[1])
#        print'  ProgressLoss:' +str(loss.data) # print every 25k frames 
    return sum_nll, sum_accuracy

# In[6]:
if __name__ == "__main__":
    ap = ArgumentParser("Semisupervised experiment")
    subparsers = ap.add_subparsers(dest='cmd', help='sub-command help')
    
    # TRAIN
    train_cmd = subparsers.add_parser('train', help='Train a new model')
    train_cmd.add_argument("save_to", type=str,   default='model', help="model filepath to save the state and results")
    train_cmd.add_argument("--ratio", type=float, default=1,       help="ratio of labeled data against unlabeled data")
    train_cmd.add_argument("--context", type=float, default=default_options_dict['context_length'],  help="length of context")
    train_cmd.add_argument("--epoch", type=int,   default=default_options_dict['n_max_epochs'], help="number of epoch")
    train_cmd.add_argument("--seed", type=int,    default=default_options_dict['rnd_seed'],    help="random seed")
    train_cmd.add_argument("--batchsize", type=int, default=default_options_dict['batch_size'],    help="batchsize")
    train_cmd.add_argument("--dataset", type=str, default='/home-nfs/tawara/work/ttic/MyPython/src/timit/timit_3.train.npz', 
                           help="Which dataset to use")
    # Analyze
    analyze_cmd = subparsers.add_parser('analyze', help='Analyze each layer')
    analyze_cmd.add_argument("load_from",   type=str, default='model',  help="Destination to load the state from")
    analyze_cmd.add_argument("--context", type=float, default=default_options_dict['context_length'],  help="length of context")
    analyze_cmd.add_argument("--data_type", type=str, default='test',   help="Data type to evaluate on")
    analyze_cmd.add_argument("--save_to",   type=str, default='noname', help="Destination to save the state and results")
    analyze_cmd.add_argument("--layer",     type=int, default=0,        help="Extract layer")
    analyze_cmd.add_argument("--batchsize", type=int, default=default_options_dict['batch_size'],    help="batchsize")
    analyze_cmd.add_argument("--dataset", type=str, default='/home-nfs/tawara/work/ttic/MyPython/src/timit/timit_3.test.npz',
                           help="Which dataset to use")
    rng = numpy.random.RandomState(1)
    args = ap.parse_args()
    t_start = time.time()

    if args.cmd == 'analyze':
        batchsize = args.batchsize
        filename = args.load_from
        print "load from %s" % filename
        model = pickle.load(open(filename,'rb'))
        model.device_id = [0]
        cuda.get_device(0).use()
        offset = range(-context_length, context_length+1)
        x_test, frame_index = load_data(\
            KaldiScp('/data2/tawara/work/ttic/MyPython/src/kaldi/timit/data/fbank/test/feats.scp'), offset)
        x_train,_ = load_data(\
            KaldiScp('/data2/tawara/work/ttic/MyPython/src/kaldi/timit/data/fbank/train_tr90/feats.scp'), \
                offsets = offset)
        N_test=x_test.shape[0]
        N_train=x_train.shape[0]
        print "Applying batch normalization"
        for i in xrange(0, N_train, batchsize):
            x_batch = x_train[i : i + batchsize]
            model.forward(x_batch,test=False)
        logger.info("Extracting final layer")
        save_to = args.save_to
        print 'Saving output layer to %s' % filename+'.post.ark'

        ark=KaldiArk(filename+'.post.ark','wb')
        for key in frame_index:
            x_batch = x_test[frame_index[key][0] : frame_index[key][1]]
            ark.write(key, cuda.to_cpu(model.forward(x_batch,test=True).data))

    elif args.cmd == "train":
        for d in option_dict:
            print d + ": " + str(option_dict[d])

        ratio   = float(args.ratio)
        n_epoch = args.epoch
        seed    = args.seed
        batchsize = args.batchsize
        model_to  = args.save_to
        numpy.random.seed(seed)
        context_length = args.context

        offset = range(-context_length, context_length+1)
        ali_to_pdf=KaldiCommand('bin/ali-to-pdf', option='/data2/tawara/work/ttic/MyPython/src/kaldi/timit/exp/tri3_ali/final.mdl')
#        feats=KaldiCommand('featbin/apply-cmvn', '--norm-means=true --norm-vars=true --utt2spk=ark:/data2/tawara/work/ttic/MyPython/src/kaldi/timit/data/fbank/train_tr90/utt2spk scp:/data2/tawara/work/ttic/MyPython/src/kaldi/timit/data/fbank/train_tr90/cmvn.scp')
        dataset = load_timit_labelled_kaldi(\
#            feats('/data2/tawara/work/ttic/MyPython/src/kaldi/timit/data/fbank/train_tr90/feats.scp'),
            KaldiArk('/data2/tawara/work/ttic/MyPython/src/kaldi/timit/feats.ark'),
                ali_to_pdf('\"gunzip -c /data2/tawara/work/ttic/MyPython/src/kaldi/timit/exp/tri3_ali/ali.*.gz |\"'), \
                offsets = offset)
        x_train, y_train = dataset

        feats=KaldiCommand('featbin/apply-cmvn', '--norm-means=true --norm-vars=true --utt2spk=ark:/data2/tawara/work/ttic/MyPython/src/kaldi/timit/data/fbank/train_cv10/utt2spk scp:/data2/tawara/work/ttic/MyPython/src/kaldi/timit/data/fbank/train_cv10/cmvn.scp')
        dataset  = load_timit_labelled_kaldi( \
            feats('/data2/tawara/work/ttic/MyPython/src/kaldi/timit/data/fbank/train_cv10/feats.scp'),
                ali_to_pdf('\"gunzip -c /data2/tawara/work/ttic/MyPython/src/kaldi/timit/exp/tri3_ali/ali.*.gz |\"'), \
                offsets = offset)
        x_valid, y_valid = dataset
        model = NN_parallel(option_dict, njobs=1)
#        model.load_parameters('/data2/tawara/work/ttic/MyPython/src/kaldi/timit/nnet.0')
        model.load_parameters('/data2/tawara/work/ttic/MyPython/src/kaldi/timit/exp/fbank/dnn4_nn5_1024_cmvn_splice10_pretrain-dbn_dnn/nnet_dbn_dnn.init')
        model.initialize()

        N_train_all = x_train.shape[0]
        indexes_l = range(N_train_all)
        indexes_ul  = list(set(range(0,N_train_all))-set(indexes_l))
        N_train_l   = len(indexes_l)
        N_train_ul  = len(indexes_ul)
        assert N_train_all == N_train_l + N_train_ul
        
        N_valid = x_valid.shape[0]
        indexes_valid = range(0, N_valid)
        for epoch in xrange(n_epoch):
            print ('[epoch '          + str(epoch) +']')
            indexes_l  = numpy.random.permutation(indexes_l)
            indexes_ul = numpy.random.permutation(indexes_ul)
            start_time = timeit.default_timer()
#            sum_nll, sum_accuracy = apply_batch(model, x_train, y_train, indexes_l, batchsize, foldings=id_mapping)
            sum_nll, sum_accuracy = apply_batch(model, x_train, y_train, indexes_l, batchsize)
            end_time = timeit.default_timer()
            print ('train nll: '       + str(sum_nll / N_train_all))
            print ("train FRAME_ACCURACY >> %f %% << (processing time: %f [sec])"  % (sum_accuracy / N_train_all * 100, end_time-start_time))
#            _, _, sum_valid_accuracy = apply_batch(x_valid, y_valid, indexes_valid, train=False)
#            logger.info('valid wer(%): '    + str(100 - sum_valid_accuracy / N_valid * 100))
            start_time = timeit.default_timer()
            _, sum_valid_accuracy  = apply_batch(model, x_valid, y_valid, indexes_valid, batchsize, train=False)
            end_time = timeit.default_timer()
            print ("valid FRAME_ACCURACY >> %f %% << (processing time: %f [sec])"  % (sum_valid_accuracy / N_valid * 100, end_time-start_time))

        filename = args.save_to
        with open(filename,'w') as f:
            pickle.dump(model,f)
