import numpy
import cPickle
import os
import gzip
import sys
import logging
logger = logging.getLogger('load_data')
from fuel.datasets.hdf5 import H5PYDataset

####################################################################

def load_swbd_dataset(setname, ratio=1.0, min_count=3):
    logger.info('loading data from %s' %setname)
    logger.info('Ratio of labeled / unlabaled: %f' % ratio)
    assert ratio > 0 or ratio <= 1
    filename = '/home/tawara/work/ttic/data/icassp15.0/' + setname + '.hdf5'
    swbd=H5PYDataset(filename, which_set=("train"),load_in_memory=True)
    x_train =swbd.data_sources[0].astype(numpy.float32)
    y_train =swbd.data_sources[1][:,0].astype(numpy.int32)
    ndata=x_train.shape[0]
    if ratio == 1.0:
        logger.info("Using all data: %d"%ndata)
        i_labeled=numpy.array(range(0,ndata))
    else:
        n_classes = y_train.max() + 1
        min_samples=sys.maxint
        max_samples=0
        avg_samples=0
        indices=numpy.array(range(0,ndata))
        i_labeled = []
        n_labeled_all = 0
        for c in range(n_classes):
            n_samples= round(sum(y_train==c) * ratio) if round(sum(y_train==c) * ratio) >= min_count else min_count
            i = (indices[y_train == c])[:n_samples]
            i_labeled += list(i)
            avg_samples += n_samples
            min_samples = n_samples if n_samples < min_samples else min_samples
            max_samples = n_samples if n_samples > max_samples else max_samples
            n_labeled_all += n_samples
        logger.info('Minimum number of labeled samples: %d' % min_samples)
        logger.info('Maximum number of labeled samples: %d' % max_samples)
        logger.info('Average number of labeled samples: %d' % int(avg_samples / n_classes))
        logger.info('Total number of labeled samples:   %d' % n_labeled_all)
        logger.info('Total number of unlabeled samples: %d' % int(ndata - n_labeled_all))

    swbd=H5PYDataset(filename, which_set=("dev"), load_in_memory=True)
    x_valid =swbd.data_sources[0].astype(numpy.float32)
    y_valid =swbd.data_sources[1][:,0].astype(numpy.int32)
    swbd=H5PYDataset(filename, which_set=("test"), load_in_memory=True)
    x_test =swbd.data_sources[0].astype(numpy.float32)
    y_test =swbd.data_sources[1][:,0].astype(numpy.int32)

    return [(x_train, y_train), (x_valid, y_valid), (x_test, y_test), i_labeled]

def load_original_swbd_dataset():
    logger.info('... loading data')
    swbd=H5PYDataset('/home-nfs/tawara/work/ttic/data/icassp15.0/swbd_recognition.hdf5',which_set=("train"),load_in_memory=True)
    x_train =swbd.data_sources[0].astype(numpy.float32)
    y_train =swbd.data_sources[1][:,0].astype(numpy.int32)
    swbd=H5PYDataset('/home-nfs/tawara/work/ttic/data/icassp15.0/swbd_recognition.hdf5',which_set=("dev"),load_in_memory=True)
    x_valid =swbd.data_sources[0].astype(numpy.float32)
    y_valid =swbd.data_sources[1][:,0].astype(numpy.int32)
    swbd=H5PYDataset('/home-nfs/tawara/work/ttic/data/icassp15.0/swbd_recognition.hdf5',which_set=("test"),load_in_memory=True)
    x_test =swbd.data_sources[0].astype(numpy.float32)
    y_test =swbd.data_sources[1][:,0].astype(numpy.int32)

    def flatten(x):
        cnt=0
        dim = x.shape
        res=numpy.zeros((dim[0], dim[1]*dim[2]*dim[3]))
        for l in x:
            res[cnt] = l.flatten()
            cnt +=1
        return res
    x_train=flatten(x_train)
    x_valid=flatten(x_valid)
    x_test =flatten(x_test)
    return ((x_train,y_train), (x_valid,y_valid), (x_test,y_test))


        
def load_swbd_for_test():
    dataset = load_original_swbd_dataset()
    
    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]

    return [(train_set_x, train_set_y), (test_set_x, test_set_y)]

"""
def load_swbd_for_validation(rng, n_l, n_v=100):
    dataset = load_original_swbd_dataset()

    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]

    _train_set_x = numpy.concatenate((train_set_x, valid_set_x), axis=0)
    _train_set_y = numpy.concatenate((train_set_y, valid_set_y), axis=0)
    rand_ind = rng.permutation(_train_set_x.shape[0])
    _train_set_x = _train_set_x[rand_ind]
    _train_set_y = _train_set_y[rand_ind]

    s_c = n_l / 10.0
    train_set_x = numpy.zeros((n_l, _train_set_x.shape[1]))
    train_set_y = numpy.zeros(n_l)
    for i in xrange(10):
        ind = numpy.where(_train_set_y == i)[0]
        train_set_x[i * s_c:(i + 1) * s_c, :] = _train_set_x[ind[0:s_c], :]
        train_set_y[i * s_c:(i + 1) * s_c] = _train_set_y[ind[0:s_c]]
        _train_set_x = numpy.delete(_train_set_x, ind[0:s_c], 0)
        _train_set_y = numpy.delete(_train_set_y, ind[0:s_c])

    print rand_ind
    rand_ind = rng.permutation(train_set_x.shape[0])
    train_set_x = train_set_x[rand_ind]
    train_set_y = train_set_y[rand_ind]
    valid_set_x = _train_set_x[:n_v]
    valid_set_y = _train_set_y[:n_v]
    # ul_train_set_x = _train_set_x[n_v:]
    train_set_ul_x = numpy.concatenate((train_set_x, _train_set_x[n_v:]), axis=0)
    train_set_ul_x = train_set_ul_x[rng.permutation(train_set_ul_x.shape[0])]
    ul_train_set_y = _train_set_y[n_v:]  # dummy

    train_set_x, train_set_y = _shared_dataset((train_set_x, train_set_y))
    train_set_ul_x, ul_train_set_y = _shared_dataset((train_set_ul_x, ul_train_set_y))
    valid_set_x, valid_set_y = _shared_dataset((valid_set_x, valid_set_y))

    return [(train_set_x, train_set_y, train_set_ul_x), (valid_set_x, valid_set_y)]
"""    
####################################################################
