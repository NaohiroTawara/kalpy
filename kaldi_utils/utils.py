import csv
import pickle
from kaldi.data import load_timit_labelled_kaldi, load_data
import numpy as np

def convert(y):
    with open('timit/triplets') as cv:
        tbl = {i:[int(p),int(s)] for p,s,i,_ in csv.reader(cv,delimiter=' ')}

    phones=[]
    states=[]

    for value in y:
        phones.append(tbl[str(value)][0])

    tmp=[]
    for value in y:
        tmp.append(str(tbl[str(value)][0])+'_'+str(tbl[str(value)][1]))
    d={}
    cnt =0
    for value in tmp:
        if not d.has_key(value):
            d[value] = cnt
            cnt += 1
        states.append(d[value])
    res={}
    res['phones']=np.asarray(phones, dtype=np.int32)
    res['states']=np.asarray(states, dtype=np.int32)
    return res

if __name__ == "__main__":
    
    x_train_lb, y_train_lb = load_timit_labelled_kaldi('fbank/train_tr90_lb10', 'models/pdf.ark', nnet_transf = 'models/final.feature_transform')
    print convert(y_train_lb)['phones'][0:10]
    print convert(y_train_lb)['states'][0:10]
    print len(set(convert(y_train_lb)['phones'])),len(set(convert(y_train_lb)['states']))
