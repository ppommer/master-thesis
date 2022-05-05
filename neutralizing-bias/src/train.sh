python joint/train.py \
       --train /home/rpryzant/bias/data/v6/corpus.wordbiased.tag.train \
       --test /home/rpryzant/bias/data/v6/corpus.wordbiased.tag.test \
       --pretrain_data /home/rpryzant/bias/data/v6/corpus.unbiased.shuf \
       --extra_features_top --pre_enrich --activation_hidden --tagging_pretrain_epochs 3 \
       --pretrain_epochs 4 --learning_rate 0.0003 --epochs 20 --hidden_size 512 --train_batch_size 24 \
       --test_batch_size 16 --bert_full_embeddings --debias_weight 1.3 --freeze_tagger --token_softmax \
       --working_dir public_model/ --pointer_generator --coverage
