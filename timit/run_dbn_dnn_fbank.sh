#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# Modified by Tetsuji Ogawa, 2014-10-18

# This example script trains a DNN on top of fMLLR features. 
# The training is done in 3 stages,
#
# 1) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs, 
#    a good starting point for frame cross-entropy trainig.
# 2) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.
# 3) sequence-training optimizing sMBR: 
#    the objective is to emphasize state-sequences with better 
#    frame accuracy w.r.t. reference alignment.

#kaldi=$1
#export KALDI_ROOT=$kaldi

depth=5
hiddim=1024
splice=10

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

# Begin of configulation
gmmdir=exp/tri3
feat=fbank
data=data/$feat
stage=0 # resume training with --stage=N
# End of configulation
. utils/parse_options.sh || exit 1;
#

exp=exp/$feat
if [ $stage -le 1 ]; then
    # Store fbank features, so we can train on them easily,
    for x in train dev test; do
	dir=$data/$x
	utils/copy_data_dir.sh data/$x $dir || exit 1;
	steps/make_fbank.sh --cmd "$train_cmd" --compress false --nj 10 $dir exp/make_fbank/$x $dir/data || exit 1;
	steps/compute_cmvn_stats.sh $dir exp/make_fbank/$x $dir/data || exit 1;
    done
    dir=$data/train
    # split the data : 90% train 10% cross-validation (held-out)
    utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi


if [ $stage -le 1 ]; then
    # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
    #dir=exp/$feat/dnn4_pretrain-dbn
    dir=$exp/dnn4_nn${depth}_${hiddim}_cmvn_splice${splice}_pretrain-dbn
    (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
    $cuda_cmd $dir/log/pretrain_dbn.log \
        steps/nnet/pretrain_dbn.sh --hid-dim $hiddim --nn-depth $depth --rbm-iter 20 \
        --splice $splice \
	--cmvn-opts "--norm-means=true --norm-vars=true" \
        $data/train $dir || exit 1;
#    $cuda_cmd $dir/log/pretrain_dbn.log \
#	steps/nnet/pretrain_dbn.sh --hid-dim 1024 --rbm-iter 20 $data/train $dir || exit 1;
fi

if [ $stage -le 2 ]; then
    # Train the DNN optimizing per-frame cross-entropy.
    dir=$exp/dnn4_nn${depth}_${hiddim}_cmvn_splice${splice}_pretrain-dbn_dnn
    ali=${gmmdir}_ali
    feature_transform=$exp/dnn4_nn${depth}_${hiddim}_cmvn_splice${splice}_pretrain-dbn/final.feature_transform
    dbn=$exp/dnn4_nn${depth}_${hiddim}_cmvn_splice${splice}_pretrain-dbn/${depth}.dbn
    (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
    # Train
    $cuda_cmd $dir/log/train_nnet.log \
        steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 \
	--hid-dim $hiddim --learn-rate 0.008 \
	--cmvn-opts "--norm-means=true --norm-vars=true" \
        --splice $splice $data/train_tr90 $data/train_cv10 data/lang $ali $ali $dir || exit 1;

    # Decode (reuse HCLG graph)
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 \
	$gmmdir/graph $data/test $dir/decode_test || exit 1;
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 \
	$gmmdir/graph $data/dev $dir/decode_dev || exit 1;
fi

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
