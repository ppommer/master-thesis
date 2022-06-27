BASE_DIR=inference/concurrent
INPUT=bias_data/WNC/biased.word.test
OUTPUT=inference/concurrent/results_concurrent.txt

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
    --out_file $BASE_DIR/output_concurrent.txt
