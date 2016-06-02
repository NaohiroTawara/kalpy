import h5py
from fuel.datasets.hdf5 import H5PYDataset

def save(filename, train_x, train_labels, dev_x, dev_labels, test_x, test_labels):
    train_len=len(train_x)
    dev_len=len(dev_x)
    test_len=len(test_x)
    dim, ndata = train_x[0].shape
    all_len = train_len + dev_len + test_len
    f = h5py.File(filename, mode='w')
    features = f.create_dataset('features', (all_len, 1, dim, ndata), dtype='float32')
    targets  = f.create_dataset('targets', (all_len, 1), dtype='uint32')
    features[...] = np.vstack([concat(train_x), concat(dev_x), concat(test_x)])
    targets[...]  = np.vstack([np.array([train_labels]).T, np.array([dev_labels]).T, np.array([test_labels]).T])
%    features.dims[0].label = 'batch'
%    features.dims[1].label = 'channel'
%    features.dims[2].label = 'height'
%    features.dims[3].label = 'width'
%    targets.dims[0].label  = 'batch'
%    targets.dims[1].label  = 'index'
    split_dict = {
        'train': {'features': (0, train_len),
                  'targets':  (0, train_len)},
        'dev'  : {'features': (train_len, train_len + dev_len),
                  'targets':  (train_len, train_len + dev_len)},
        'test' : {'features': (train_len + dev_len, train_len + dev_len + test_len),
                  'targets':  (train_len + dev_len, train_len + dev_len + test_len)}}
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()
