# coding: utf-8

# In[1]:
import sys
sys.path.append('../../')

from chainer import Variable, optimizers,cuda
import cupy
from swbd.load_data import load_swbd_dataset

import logging

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
import chainer.computational_graph as C

from scipy.spatial.distance import pdist
from swbd.data_io import swbd_utts_to_labels
import samediff

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
ch=logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

logger = logging.getLogger(__name__)
default_options_dict = {
    "njobs": 1,
    "rnd_seed": 42,
    "batch_size": 22,
    "n_max_epochs": 20,
    "model_dir": "models",
    "online_evaluation": False,
    "layers_specs": [
        {"type": "conv", "filter_shape": (96,1,39,9), "pool_shape": 3, "activation": 'relu',
         'BN': (True,True), 
#         'dropout': 0.5
        },
        {"type": "conv", "filter_shape": (96,96,1,8), "pool_shape": 3, "activation": 'relu',
         'BN': (True,True),  # apply BN after relu and pooling layers
#         'dropout': 0.5
        },
        {"type": "full", "dimensions": (1824, 1024), "activation": 'relu', 
         'BN': (True), 
#         'dropout': 0.5
        },
        {"type": "full", "dimensions": (1024, 512), "activation": 'relu', 
         'BN': (True), 
#         'dropout': 0.5
        },
        {"type": "full", "dimensions": (512, 1061), "activation": 'linear', 
         'BN': (True), 
#         'dropout': 0.5
        },
    ],
#    "optimizer":optimizers.SGD(0.008),
    "optimizer":optimizers.Adam(alpha=0.002),
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

def apply_batch(model, x_train, y_train, indexes, batchsize, train = True, finetune = False):
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
        bs = x_batch.shape[0]
#        output  = model.forward(x_batch, test=True)
        if train:
#            diff = calc_diff(F.softmax(output), y_batch, y_batch.shape[0]).data
            loss = model.loss_softmax(x_batch, y_batch, test=test, finetune=finetune)
            sum_nll  += cuda.to_cpu(loss.data) * bs
#            loss = nll # Maximum Likelihood Estimation
            model.zero_grads()
            loss.backward()
            model.update()
        sum_accuracy  += F.accuracy(model.forward(x_batch, test=True), \
                                        Variable(cuda.to_gpu(y_batch,device=model.device_id[0]))).data * bs

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
    train_cmd.add_argument("--epoch", type=int,   default=default_options_dict['n_max_epochs'], help="number of epoch")
    train_cmd.add_argument("--seed", type=int,    default=default_options_dict['rnd_seed'],    help="random seed")
    train_cmd.add_argument("--batchsize", type=int, default=default_options_dict['batch_size'],    help="batchsize")
    train_cmd.add_argument("--dataset", type=str, default='swbd_recognition',
                           choices=['swbd_recognition', 'swbd_verification', 'swbd_short'], help="Which dataset to use")

    # Analyze
    analyze_cmd = subparsers.add_parser('analyze', help='Analyze each layer')
    analyze_cmd.add_argument("load_from",   type=str, default='model',  help="Destination to load the state from")
    analyze_cmd.add_argument("--data_type", type=str, default='test',   help="Data type to evaluate on")
    analyze_cmd.add_argument("--save_to",   type=str, default='noname', help="Destination to save the state and results")
    analyze_cmd.add_argument("--layer",     type=int, default=0,        help="Extract layer")
    analyze_cmd.add_argument("--batchsize", type=int, default=default_options_dict['batch_size'],    help="batchsize")
    analyze_cmd.add_argument("--dataset", type=str, default='swbd_verification',
                           choices=['swbd_recognition', 'swbd_verification', 'swbd_short'], help="Which dataset to use")
    rng = numpy.random.RandomState(1)
    args = ap.parse_args()
    t_start = time.time()

    if args.cmd == 'analyze':
        batchsize = args.batchsize
        filename = args.load_from
        print "load from %s" % filename
#        model = pickle.load(open(filename,'rb'))
        model = NN_parallel({}, njobs=1)
        model.load(filename)
        npz_tst = numpy.load("/data2/tawara/work/ttic/data/icassp15.0/swbd.test.npz")
        npz_trn = numpy.load("/data2/tawara/work/ttic/data/icassp15.0/swbd.train.npz")
        utt_ids_trn = sorted(npz_trn.keys())
        utt_ids_tst = sorted(npz_tst.keys())
        x_train = numpy.array([npz_trn[i] for i in utt_ids_trn]).astype(numpy.float32)
        x_test  = numpy.array([npz_tst[i] for i in utt_ids_tst]).astype(numpy.float32)
        tmp=[]
        for l in x_test:
            tmp.append([l])
        x_test=numpy.array(tmp)
        tmp=[]
        for l in x_train:
            tmp.append([l])
        x_train=numpy.array(tmp)
        N_test=x_test.shape[0]
        N_train=x_train.shape[0]
        batchsize=22
        logger.info("Applying batch normalization")
        for i in xrange(0, N_train, batchsize):
            x_batch = x_train[i : i + batchsize]
            model.forward(x_batch,test=False)
        logger.info("Extracting final layer")
        save_to = args.save_to
        X=[]
        for i in xrange(0, N_test):
            utt_id = utt_ids_tst[i]
            x_batch = x_test[i : i + 1]
            X.append(cuda.to_cpu(F.softmax(model.forward(x_batch,test=True)).data))
        X=numpy.asarray(X)[:,0,:]
        logger.info("Calcurating average precision")
        start_time = timeit.default_timer()
        labels = swbd_utts_to_labels(utt_ids_tst)
        distances = pdist(X, metric="cosine")
        matches = samediff.generate_matches_array(labels)
        ap, prb = samediff.average_precision(distances[matches == True], distances[matches == False])
        end_time = timeit.default_timer()
        logger.info("Average precision: %s (processing time: %f [sec])"  % (str(ap), end_time-start_time))

        logger.info('Saving output layer to %s' % save_to+".npz")
        numpy.savez_compressed(save_to, X)

#        dataset = load_swbd_dataset(args.dataset, ratio=1)  
#       x_train, y_train = dataset[0] 
#        x_test, y_test   = dataset[2]
#        N_test=x_test.shape[0]
#        N_train=x_train.shape[0]
#        print "Applying batch normalization"
#        for i in xrange(0, N_train, batchsize):
#            x_batch = x_train[i : i + batchsize]
#            model.forward(x_batch,test=False)
#        logger.info("Extracting final layer")
#        save_to = args.save_to
#        print 'Saving output layer to %s' % filename



    elif args.cmd == "train":
        for d in option_dict:
            print d + ": " + str(option_dict[d])

        ratio   = float(args.ratio)
        n_epoch = args.epoch
        seed    = args.seed
        batchsize = args.batchsize
        model_to  = args.save_to
        numpy.random.seed(seed)

        dataset = load_swbd_dataset(args.dataset, ratio=ratio)
        x_train, y_train = dataset[0]
        x_valid, y_valid = dataset[1]
        x_test, y_test   = dataset[2] 
        indexes_l        = dataset[3]
        model = NN_parallel(option_dict, njobs=1)
        model.initOptimizer()

        N_train_all = x_train.shape[0]
        indexes_l   = range(N_train_all)
        indexes_ul  = list(set(range(0,N_train_all))-set(indexes_l))
        N_train_l   = len(indexes_l)
        N_train_ul  = len(indexes_ul)
        assert N_train_all == N_train_l + N_train_ul
        
        N_valid = x_valid.shape[0]
        N_test = x_test.shape[0]
        indexes_valid = range(0, N_valid)
        indexes_test   = range(0, N_test)

        for epoch in xrange(n_epoch):
            print ('[epoch '          + str(epoch) +']')
            indexes_l  = numpy.random.permutation(indexes_l)
            indexes_ul = numpy.random.permutation(indexes_ul)
            start_time = timeit.default_timer()
            sum_nll, sum_accuracy = apply_batch(model, x_train, y_train, indexes_l, batchsize)
            end_time = timeit.default_timer()
            print ('train nll: '       + str(sum_nll / N_train_all))
            print ("train WORD_ERR >> %f %% << (processing time: %f [sec])"  % (100-sum_accuracy / N_train_all * 100, end_time-start_time))
            start_time = timeit.default_timer()
            _, sum_valid_accuracy  = apply_batch(model, x_valid, y_valid, indexes_valid, batchsize, train=False)
            end_time = timeit.default_timer()
            print ("valid WORD_ERR >> %f %% << (processing time: %f [sec])"  % (100-sum_valid_accuracy / N_valid * 100, end_time-start_time))
            start_time = timeit.default_timer()
            _, sum_test_accuracy  = apply_batch(model, x_test, y_test, indexes_test, batchsize, train=False)
            end_time = timeit.default_timer()
            print ("test WORD_ERR >> %f %% << (processing time: %f [sec])"  % (100-sum_test_accuracy / N_test * 100, end_time-start_time))
            model.save(args.save_to+"_"+str(epoch))
