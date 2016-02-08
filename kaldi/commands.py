import os
import subprocess
import struct
import kaldi.io
from kaldi.io import KaldiArkPipe,KaldiArk
import io
"""
Following functions call Kaldi commands,
so Kaldi (https://github.com/kaldi-asr/kaldi) must be installed on $KALDI_PATH
"""
class KaldiCommand:
    def __init__(self, cmd, option='', KALDI_PATH=None):
        self.__name = cmd
        self.setOption(option)

        if KALDI_PATH is None:
            assert os.environ.has_key('KALDI_PATH'), 'Cannot find environment variable $KALDI_PATH'
            KALDI_PATH = os.environ['KALDI_PATH']
        self.cmd_path = KALDI_PATH + '/src/' + cmd
        assert os.path.exists(self.cmd_path), self.cmd_path + \
            ': No such Kaldi command.\n' \
            'Directory that contains kaldi command is also required as well as command name (e.x. featbin/copy-feats).'

    def __str__(self):
        return self.cmd_path +'\nCommand: ' + self.__name  + '\nOption:  ' + self.__option
    
    def __call__(self, d, binary=True):
        return self.run(d, binary)

    def setOption(self, option):
        self.__option = option

    def __lshift__(self, d):
        '''
        Writer: self << d
        e.x. KaldiCommand("featbin/splice-feats") << KaldiCommand("featbin/add-deltas") << KaldiArk('tmp.ark')
        '''
        return self(d)

    def __call__(self, d, binary=True):
        if type(d) == str:
            return self.run_file(d,binary)
        else:
            return self.run(d,binary)

    def run(self, d, binary=True):
        #Execute Kaldi command inputting dict d and return KaldiFormat handler
        ark_in  = KaldiArk('/tmp/pipe.ark','wb')
        for key, mat in d:
            ark_in.write(key, mat)
        ark_in.p.close()
        return self.run_file('/tmp/pipe.ark', binary)

#    def run(self, d, binary=True):
#        opt = '' if binary else ',t'
#        cmd = self.cmd_path + " ark:-" + " ark" + opt + ":-"
#        p = subprocess.Popen(cmd.split(' '),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT, shell=False,bufsize=100000000)
#        ark_in  = KaldiArkPipe(p.stdin)
        # Following part causes dead lock when buffer of pipe is exhausted.
#        for key, mat in d:
#            print key
#            ark_in.write(key, mat)
#        ark_in.p.close()
#        return KaldiArkPipe(p.stdout)

    def run_file(self, fn_in, binary=True):
        '''
        Execute Kaldi command inputting dict d and return KaldiFormat handler
        '''
        ext = os.path.splitext(fn_in)[-1][1:]
        ext = 'scp' if ext =='scp' else 'ark'
        assert ext == 'scp' or ext == 'ark', 'Unsupported file format: ' + ext    
        opt = '' if binary else ',t'
        if self.__option is '':
            cmd = self.cmd_path  + ' ' +ext + ":" + fn_in + " ark" + opt + ":-"
        else:
            cmd = self.cmd_path + ' ' +self.__option + ' ' +ext + ":" + fn_in + " ark" + opt + ":-"
        print '# '+cmd
        p = subprocess.Popen(cmd,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        return KaldiArkPipe(p.stdout)


import sys
# Debug and test
if __name__ == '__main__':
    copy_feats=KaldiCommand('featbin/copy-feats')
    filename= '/data2/tawara/work/ttic/MyPython/src/kaldi/timit/data/fbank/test/data/raw_fbank_test.10.ark'
    ark = copy_feats << filename
    d1={key:mat for key,mat in ark }
    print str(len(d1)) + ' files are read.'


