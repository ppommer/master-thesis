#!/bin/sh

FOLDERNAME=$1
ROBERTA_LARGE="/home/ppommer/repos/master-thesis/style-transfer-paraphrase/models/roberta.large"

python /home/ppommer/repos/master-thesis/style-transfer-paraphrase/fairseq/preprocess.py \
    --only-source \
    --trainpref "${FOLDERNAME}/train.label" \
    --validpref "${FOLDERNAME}/dev.label" \
    --destdir "${FOLDERNAME}-bin/label" \
    --workers 24

python /home/ppommer/repos/master-thesis/style-transfer-paraphrase/fairseq/preprocess.py \
    --only-source \
    --trainpref "${FOLDERNAME}/train.input0.bpe" \
    --validpref "${FOLDERNAME}/dev.input0.bpe" \
    --destdir "${FOLDERNAME}-bin/input0" \
    --workers 24 \
    --srcdict $ROBERTA_LARGE/dict.txt

cp ${FOLDERNAME}-bin/label/dict.txt ${FOLDERNAME}-bin/dict.txt
cp ${FOLDERNAME}-bin/label/dict.txt ${FOLDERNAME}/dict.txt
