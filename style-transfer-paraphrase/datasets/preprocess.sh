#!/bin/sh

FOLDERNAME=$1
ROBERTA_LARGE="/home/pp/master-thesis/style-transfer-paraphrase/models/roberta.large"

# Convert plaintext dataset into it's BPE form
python datasets/dataset2bpe.py --dataset "datasets/${FOLDERNAME}"

# Convert the BPE codes to fairseq binaries and build a label dictionary
python ~/master-thesis/style-transfer-paraphrase/fairseq/preprocess.py \
    --only-source \
    --trainpref "${FOLDERNAME}/train.label" \
    --validpref "${FOLDERNAME}/dev.label" \
    --destdir "${FOLDERNAME}-bin/label" \
    --workers 24

python ~/master-thesis/style-transfer-paraphrase/fairseq/preprocess.py \
    --only-source \
    --trainpref "${FOLDERNAME}/train.input0.bpe" \
    --validpref "${FOLDERNAME}/dev.input0.bpe" \
    --destdir "${FOLDERNAME}-bin/input0" \
    --workers 24 \
    --srcdict $ROBERTA_LARGE/dict.txt

cp ${FOLDERNAME}-bin/label/dict.txt ${FOLDERNAME}-bin/dict.txt
cp ${FOLDERNAME}-bin/label/dict.txt ${FOLDERNAME}/dict.txt

# Paraphrase the dataset
python datasets/paraphrase_splits.py --dataset "datasets/${FOLDERNAME}"
