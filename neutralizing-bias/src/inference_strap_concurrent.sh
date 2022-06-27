DATA=/home/ppommer/repos/master-thesis/style-transfer-paraphrase/inference/strap_large/output_strap_large_6.txt
BASE_DIR=inference/strap_concurrent

INPUT=$BASE_DIR/input.txt
OUTPUT=$BASE_DIR/results_strap_concurrent.txt

python utils/prepare_strap.py \
    --input $DATA \
    --output $INPUT \
    --gold bias_data/WNC/biased.word.test

python joint/inference.py \
    --test $INPUT \
    --inference_output $OUTPUT \
    --working_dir $BASE_DIR \
    --bert_encoder \
    --bert_full_embeddings \
    --coverage \
    --debias_checkpoint models/concurrent.ckpt \
    --debias_weight 1.3 \
    --no_tok_enrich \
    --pointer_generator \
    --test_batch_size 1

python utils/generate_output.py \
    --in_file $OUTPUT \
    --out_file $BASE_DIR/output_strap_concurrent.txt

rm $BASE_DIR/input.txt
