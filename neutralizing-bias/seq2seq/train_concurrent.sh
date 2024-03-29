WNC=../data/WNC
WORKING_DIR=../models

python train.py \
       --train $WNC/biased.word.train \
       --test $WNC/biased.word.test \
       --pretrain_data $WNC/neutral \
       --bert_full_embeddings \
       --bert_encoder \
       --debias_weight 1.3 \
       --pointer_generator \
       --coverage \
       --no_tok_enrich \
       --working_dir $WORKING_DIR
