BASE_DIR=inference/concurrent

INPUT_SINGLE=bias_data/WNC_edit/single_test.txt
OUTPUT_SINGLE=$BASE_DIR/results_concurrent_single.txt
INPUT_MULTI=bias_data/WNC_edit/multi_test.txt
OUTPUT_MULTI=$BASE_DIR/results_concurrent_multi.txt

python joint/inference.py \
    --test $INPUT_SINGLE \
    --inference_output $OUTPUT_SINGLE \
    --working_dir $BASE_DIR \
    --bert_encoder \
    --bert_full_embeddings \
    --coverage \
    --debias_checkpoint models/concurrent.ckpt \
    --debias_weight 1.3 \
    --no_tok_enrich \
    --pointer_generator \
    --test_batch_size 1

python joint/inference.py \
    --test $INPUT_MULTI \
    --inference_output $OUTPUT_MULTI \
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
    --in_file $OUTPUT_SINGLE \
    --out_file $BASE_DIR/output_concurrent_single.txt

python utils/generate_output.py \
    --in_file $OUTPUT_MULTI \
    --out_file $BASE_DIR/output_concurrent_multi.txt
