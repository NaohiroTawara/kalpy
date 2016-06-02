#from  scipy.spatial import KDTree
from kaldi.io import KaldiArk
from sklearn.neighbors import KDTree
import numpy as np


if __name__ -- "__main__":

    in_filename=sys.argv[0]
    out_filename=sys.argv[0]
    ark = KaldiArk(in_file)

    print "Reading from" + in_filename
    d1={key: m for key,m in ark}
    print "Read " + str(len(d1)) " keys."

N=60000
x_train, x_test = np.split(mnist['data'], [N])
y_train, y_test = np.split(mnist['target'], [N])


kdt = KDTree(x_train, leaf_size=30,metric='euclidean')
res = kdt.query(x_test, k=10)
