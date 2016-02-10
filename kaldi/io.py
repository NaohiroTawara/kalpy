#!/usr/bin/env python

import gzip
import struct
import numpy as np
from os import path
import io

def smart_open(filename, mode=None):
    """Opens a file normally or using gzip based on the extension."""
    if path.splitext(filename)[-1] == ".gz":
        if mode is None:
            mode = "rb"
        return gzip.open(filename, mode)
    else:
        if mode is None:
            mode = "r"
        return open(filename, mode)

def __write_token__(p, token, argv=None):
    p.write(struct.pack('<%ds'%(len(token)+1), token+' '))
    if argv is not None:
        for value in argv:
            print type(value)
            if type(value) is int:
                p.write(struct.pack("<bi", 4, value))
            elif type(value) is float:
                p.write(struct.pack("<bf", 4, value))

def __write_matrix__(p, matrix):
    if len(matrix.shape) ==1:
        rows = matrix.shape[0]
        cols = 1
        p.write(struct.pack('<ccc', 'F','V',' '))
        p.write(struct.pack('<bi', 4, rows))
    else:
        rows,cols = matrix.shape
        p.write(struct.pack('<ccc', 'F','M',' '))
        p.write(struct.pack('<bibi', 4, rows, 4, cols))
    p.write(struct.pack('<' + 'f'*rows*cols, *matrix.flatten()))


def __write_binary_header__(p):
    p.write(struct.pack('<xc', 'B'))

def __read_data__(p):
    is_binary, c = __check_binary__(p)
    if is_binary:
        return __read_binary_data__(p)
    else:
        return __read_text_data__(p, c)

def __read_binary_data__(p):
    tmp = struct.unpack("<c", p.read(1))[0]
    assert tmp != "C", "line is compressed (not surpoted)"
    if  tmp =='F':
        return __read_binary_matrix__(p)
    else:
        return __read_binary_vector__(p,np.int32)

def __read_text_data__(p, c):
    if c[1]== '[':
        p.readline()
        tmp=[]
        while True:
            line =p.readline().split(' ')
            tmp.append(map(float,line[2:-1]))
            if line[-1] == ']\n':
                break
        return np.array(tmp, dtype=np.float32)
    else:
        return np.array([float(c[0])]+map(int,p.readline().split(' ')[0:-1]), dtype=np.int32)

def __read_binary_matrix__(p):
    """
    Read a binary Kaldi archive and return a dict of Numpy matrices
    Based on the code:
    https://github.com/yajiemiao/pdnn/blob/master/io_func/kaldi_feat.py
    Format:
     ?BFM4i4i...
      - '?' is space.
      - 'B' means this line is written in binary format
      - 'FM' means this line is Float matrix. Another choise is DV meaning Double vector.
         If the first character is 'C', this line is compressed and this function doesn't support it.
      - '4i4i' first part 4i means  number of colmuns(int) and its bitsize(4)?, and latter one means number of rows(int).
    Surported formats: *.ark, output from nnet-forward, and etc.
    """
    header = struct.unpack("<cx", p.read(2))
    assert header[0] == "M" or header[0] == "V", "Input is not matrix nor vector"
    m, rows = struct.unpack("<bi", p.read(5))
    if header[0] == "M":
        n, cols = struct.unpack("<bi", p.read(5))
    else:
        cols=1
    tmp_mat = np.frombuffer(p.read(rows*cols*4), dtype=np.float32)
    utt_mat = np.reshape(tmp_mat, (rows, cols))
    return utt_mat

def __read_binary_vector__(p, dtype):
    """
    Somehow some format (e.x. *.ali) doesn't have FM part.
    Therefore, it is impossible to decide type.
    Surported formats: *.ali, output from copy-int-vector, and etc.
    """
    cols = struct.unpack("<i", p.read(4))[0]
    vec=[]
    # np.frombuffer() is unavailable because of format
    for i in range(cols):
        if dtype==np.float32:
            vec.append(struct.unpack("<xf", p.read(5))[0])
        elif dtype==np.int32:
            vec.append(struct.unpack("<xi", p.read(5))[0])
    return np.array(vec, dtype=dtype)

def __read_posterior__(p):
    # format is ; [id, posterior], ...
    # e.x. fdhc0_si1559 [3, 1.0], [2, 1.0] ... 
    if __check_binary__(p)[0]:
        n, rows = struct.unpack("<bi", p.read(5))
        n, cols = struct.unpack("<bi", p.read(5))
        vec=[]
        for i in range(rows):
            posteriors=[]
            while True:
                posteriors.append(struct.unpack("<xixf", p.read(10)))
                if struct.unpack("<xi", p.read(5))[0]  == 1 :
                    break
            vec.append(posteriors)
        return vec
    else:
        return np.array(map(int,p.readline().split(' ' )[0:-1]), dtype=np.int32)

def __check_binary__(p):
    header = struct.unpack("<cc", p.read(2))
    if header[1] == "B" :
        return True, header
    else:
        return False, header

# Following function is efficient but p.seek is unavilable if p is stdout.
'''
def __read_next_key__(p, bufsize):
    buf = p.read(bufsize)
    bufsize=len(buf)
    pos = buf.find(' ')
    token = buf[0:pos]
    if len(token) == 0:
        raise StopIteration()
    p.seek(pos-bufsize+1,1)
    return token
'''

def __read_next_key__(p, bufsize):
    token =''
    cnt = 0
    while True:
        buf = p.read(1)
        if buf ==' ':
            break
        if cnt > bufsize:
            raise StopIteration()
        token += buf
        cnt += 1
    return token

class KaldiFormat(object):
    def __init__(self, filename, mode):
        self.filename = filename
        self.p = smart_open(filename, mode)
        self.cnt = 0
    def __del__(self):
        self.p.close()
    def __str__(self):
        return self.filename
    def __iter__(self):
        return self
    def reset(self):
        assert self.p.name != '<fdopen>', 'Reset is prohibited on handler of pipeline.'
        self.p.seek(0)
        self.cnt = 0

class KaldiPosterior(KaldiFormat):
    def __init__(self, filename, mode =None):
        super(KaldiPosterior, self).__init__(filename, mode)

    def next(self, bufsize=32):
        token = __read_next_key__(self.p, bufsize)
        self.cnt += 1
        return token, __read_posterior__(self.p)

class KaldiAlignment(KaldiFormat):
    """
    Read a binary or plane Kaldi vector and return a dict of Numpy matrices, with the
    utterance IDs of the SCP as keys.
    """
    def __init__(self, filename, mode =None):
        super(KaldiAlignment, self).__init__(filename, mode)

    def write(utt_id, utt_mat):
        """
        Flush a binary of Kaldi archive to pipline
        """
        rows = utt_mat.shape
        self.p.write(struct.pack('<%ds'%(len(utt_id)), utt_id))
        self.p.write(struct.pack('<cxcc', ' ','B',' '))
        self.p.write(struct.pack('<bi', 4, rows))
        self.p.write(struct.pack('<' + 'f'*rows, *utt_mat.flatten()))

    def next(self, bufsize=32):
        token = __read_next_key__(self.p, bufsize)
        self.cnt += 1
        return token, __read_data__(self.p)

class KaldiScp(KaldiFormat):
    """
    Read a Kaldi script and return a dict of Numpy matrices, with the
    utterance IDs of the SCP as keys
    """
    def __init__(self, filename, mode =None):
        super(KaldiScp, self).__init__(filename, mode)

    def next(self):
        line = self.p.readline()
        if len(line) == 0:
            raise StopIteration()
        key, path_pos = line.replace("\n", "").split(" ")
        ark_path, pos = path_pos.split(":")
        ark_read_buffer = smart_open(ark_path, "rb")
        ark_read_buffer.seek(int(pos),0)        
        mat = __read_data__(ark_read_buffer)
        ark_read_buffer.close()
        self.cnt += 1
        return key, mat

class KaldiArk(KaldiFormat):
    """
    Read a binary or plane Kaldi archive and return a dict of Numpy matrices, with the
    utterance IDs of the SCP as keys.
    """
    def __init__(self, filename, mode =None):
        super(KaldiArk, self).__init__(filename, mode)

    def write(self, utt_id, utt_mat):
        """
        Flush a binary Kaldi archive to pipline
        """
        rows, cols = utt_mat.shape
        self.p.write(struct.pack('<%ds'%(len(utt_id)), utt_id))
        self.p.write(struct.pack('<cxcccc', ' ','B','F','M',' '))
        self.p.write(struct.pack('<bi', 4, rows))
        self.p.write(struct.pack('<bi', 4, cols))
        self.p.write(struct.pack('<' + 'f'*rows*cols, *utt_mat.flatten()))

    def next(self, bufsize=32):
        token = __read_next_key__(self.p, bufsize)
        self.cnt += 1
        return token, __read_data__(self.p)


class KaldiArkPipe(KaldiArk):
    def __init__(self, p):
        self.p = p
        self.cnt = 0

class KaldiNnetComponent:
    def __init__(self, p):
        self.p = p        
    def __iter__(self):
        return self
    def get(self):
        func, dim = self.next()
        conf ={}
        if func == 'AffineTransform':
            conf = {att: argv for att, argv in self}
            conf['W'] = __read_binary_data__(self.p)
            conf['b'] = __read_binary_data__(self.p)
        elif func == 'Splice':
            vec=[]
            for i in range(dim[-1]):
                vec.append(struct.unpack("<i", self.p.read(4))[0])
            conf['context'] = vec
        elif func == 'AddShift' or func=='Rescale':
            f, val=self.next()
            conf[f]=val
            conf['b'] = __read_binary_data__(self.p).T            
        conf['func']=func
        conf['dim']=dim
        return conf
    def next(self, bufsize=32):
        buf = self.p.read(bufsize)
        bufsize=len(buf)
        pos = buf.find('>')
        if buf.find('>')==-1 or buf.find('<')!=0:
            self.p.seek(-bufsize,1)
            raise StopIteration()
        token = buf[buf.find('<')+1:pos]
        if token == 'Nnet' or token == '/Nnet':
            self.p.seek(pos-bufsize+2,1)
            return self.next(bufsize)
        argv=[]
        self.p.seek(pos-bufsize+2,1)
        while True:
            chunk=self.p.read(1)
            if chunk =='' or struct.unpack("<b", chunk)[0] != 4:
                break
            if token=='LearnRateCoef' or token=='BiasLearnRateCoef':
                value=struct.unpack("<f", self.p.read(4))[0]
            else:
                value=struct.unpack("<i", self.p.read(4))[0]
            argv.append(value)
        self.p.seek(-1,1)
        return token, argv



# Following functions are for Kaldi's nnet files
class KaldiNnet(KaldiFormat):
    def __init__(self, filename, mode =None):
        super(KaldiNnet, self).__init__(filename, mode)
        if mode is not 'wb':
            self.binary = __check_binary__(self.p)[0]
    
    def next(self):
        return KaldiNnetComponent(self.p).get()

    def write(self, tokens):
        '''
        Flush binary in Kaldi's nnet1 format (e.x. <Nnet> and its argv, and matrix) .
        Tokens is a list of pairs of token and corresponding values:
          -  tokens[0][0]: token (e.x. ffineTransform)
          -  tokens[0][1]: tupple of values (optional) (e.x. (400,400))
          -  If you want to write matrix, set tokens[*][0] = '**MATRIX**' and tokens[*][1] = instance of np.ndarray
        '''
        __write_binary_header__(self.p)
        for token in tokens:
            values = token[1] if len(token)>1 else None 

            if token[0] == '**MATRIX**':
                __write_matrix__(self.p, values)
            else:
                __write_token__(self.p, '<' + token[0] +'>', values)

# Debug and test
if __name__ == '__main__':
    print '=================================================='
    print '====      Testing KaldiNnet in write mode    ====='
    print '=================================================='
    cnt=0
    filename ='/tmp/tmp.nnet'
    print '\nWriting dummy nnet structure in' +filename
    mat = np.asarray(range(4),dtype=np.float32).reshape(2,2)
    vec = np.asarray(range(2),dtype=np.float32)
    tokens = [ ['Nnet'], \
                   ['AffineTransform', (2,2)], \
                   ['LearnRateCoef', (1.0,)], \
                   ['BiasLearnRateCoef', (1.0,)], \
                   ['MaxNorm', (0,)], \
                   ['**MATRIX**', mat], \
                   ['**MATRIX**', vec], \
                   ['Softmax',(2,2)], ['/Nnet'] ]
    nnet = KaldiNnet(filename,'wb')
    nnet.write(tokens)
    del nnet
    print '=================================================='
    print '====            Testing KaldiNnet            ====='
    print '=================================================='
    cnt=0
#    filename ='timit/exp/fbank/dnn4_nn5_1024_cmvn_splice0_pretrain-dbn/1.dbn'
    filename ='/tmp/tmp.nnet'
    print '\nReading from ' +filename
    for layer in KaldiNnet(filename):
        print 'Layer' + str(cnt)
        print layer
        cnt +=1

    print '=================================================='
    print '====           Testing KaldiNnet (text)      ====='
    print '=================================================='
    cnt=0
    filename ='timit/exp/fbank/dnn4_nn5_1024_cmvn_splice10_pretrain-dbn_dnn/final.nnet'
#    filename = 'timit/tmp'
    print '\nReading from ' +filename
    for layer in KaldiNnet(filename):
        print 'Layer' + str(cnt)
        print layer
        cnt +=1


    print '\n=================================================='
    print '====            Testing KaldiScp             ====='
    print '=================================================='
    filename= '/data2/tawara/work/ttic/MyPython/src/kaldi/timit/data/fbank/train/feats.scp'
    print 'Reading from ' +filename
    scp = KaldiScp(filename)
    d1={key:mat for key,mat in scp }
    print str(len(d1)) + ' files are read.'
    print d1.keys()[0]
    print d1[d1.keys()[0]]

    print '\n=================================================='
    print '====            Testing KaldiArk             ====='
    print '=================================================='
    filename= '/data2/tawara/work/ttic/MyPython/src/kaldi/timit/data/fbank/test/data/raw_fbank_test.1.ark'
    print 'Reading from ' +filename
    ark = KaldiArk(filename)
    d1={key:mat for key,mat in ark }
    print str(len(d1)) + ' files are read.'
    print d1.keys()[0]
    print d1[d1.keys()[0]]

    print '\n=================================================='
    print '====       Testing KaldiArk(alignment)       ====='
    print '=================================================='
    from kaldi.commands import KaldiCommand
    filename= '\"gunzip -c timit/exp/tri3_ali/ali.*.gz |\"'
    print 'Reading from ' +filename
    ali_to_pdf=KaldiCommand('bin/ali-to-pdf',  'timit/exp/tri3_ali/final.mdl')
    pdf = ali_to_pdf << filename
    d1={key:mat for key,mat in pdf}
    print str(len(d1)) + ' files are read.'
    print d1.keys()[0]
    print d1[d1.keys()[0]]

#    print '\n=================================================='
#    print '====  Testing KaldiArk(alignment, text mode) ====='
#    print '=================================================='
#    filename= 'text.ali'
#    print 'Reading from ' +filename
#    ark = KaldiArk(filename)
#    d1={key:mat for key,mat in ark }
#    print str(len(d1)) + ' files are read.'
#    print d1.keys()[0]
#    print d1[d1.keys()[0]]

    print '\n=================================================='
    print '====       Testing KaldiArk(text mode)       ====='
    print '=================================================='
    from kaldi.commands import KaldiCommand
    copy_feats=KaldiCommand('featbin/copy-feats')
    filename= '/data2/tawara/work/ttic/MyPython/src/kaldi/timit/data/fbank/test/data/raw_fbank_test.1.ark'
    ark = copy_feats(filename, binary=False)
    d1={key:mat for key,mat in ark }
    print str(len(d1)) + ' files are read.'
    print d1.keys()[0]
    print d1[d1.keys()[0]]

    print '\nAll tests are completed.' 
