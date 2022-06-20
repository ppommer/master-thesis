DATA=/home/ppommer/repos/master-thesis/style-transfer-paraphrase/inference/WNC_large/output_strap_large_6.txt
BASE_DIR=/home/ppommer/repos/master-thesis/neutralizing-bias/src/inference/modular/strap_modular

python prepare_strap.py \
    --input $DATA \
    --output $BASE_DIR/input_strap_modular.txt

python joint/inference.py \
    --test $BASE_DIR/input_strap_modular.txt \
    --inference_output $BASE_DIR/results_strap_modular.txt \
    --activation_hidden \
    --bert_full_embeddings \
    --checkpoint models/modular.ckpt \
    --coverage --debias_weight 1.3 \
    --extra_features_top \
    --pointer_generator \
    --pre_enrich \
    --token_softmax \
    --working_dir inference_modular/
