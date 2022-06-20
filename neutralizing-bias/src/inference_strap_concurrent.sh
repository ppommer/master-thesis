DATA=/home/ppommer/repos/master-thesis/style-transfer-paraphrase/inference/WNC_large/output_strap_large_6.txt
BASE_DIR=/home/ppommer/repos/master-thesis/neutralizing-bias/src/inference/concurrent/strap_concurrent

python prepare_strap.py \
    --input $DATA \
    --output $BASE_DIR/input_strap_concurrent.txt

python joint/inference.py \
    --test $BASE_DIR/input_strap_concurrent.txt \
    --inference_output $BASE_DIR/results_strap_concurrent.txt \
    --bert_encoder \
    --bert_full_embeddings \
    --coverage \
    --debias_checkpoint models/concurrent.ckpt \
    --debias_weight 1.3 \
    --no_tok_enrich \
    --pointer_generator \
    --working_dir inference_concurrent/
