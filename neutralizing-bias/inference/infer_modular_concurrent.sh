BASE_DIR=modular_concurrent
MODEL_DIR=../models/concurrent.ckpt

INPUT_SINGLE=modular/results_modular_single.txt
INPUT_MULTI=modular/results_modular_multi.txt

TEST_SINGLE=$BASE_DIR/input_single.txt
TEST_MULTI=$BASE_DIR/input_multi.txt

INFERENCE_OUTPUT_SINGLE=$BASE_DIR/results_modular_concurrent_single.txt
INFERENCE_OUTPUT_MULTI=$BASE_DIR/results_modular_concurrent_multi.txt

python ../utils/prepare_next.py \
    --input $INPUT_SINGLE \
    --output $TEST_SINGLE

python ../utils/prepare_next.py \
    --input $INPUT_MULTI \
    --output $TEST_MULTI

python ../joint/inference.py \
    --test $TEST_SINGLE \
    --inference_output $INFERENCE_OUTPUT_SINGLE \
    --working_dir $BASE_DIR \
    --bert_encoder \
    --bert_full_embeddings \
    --coverage \
    --debias_checkpoint $MODEL_DIR \
    --debias_weight 1.3 \
    --no_tok_enrich \
    --pointer_generator \
    --test_batch_size 1

python ../joint/inference.py \
    --test $TEST_MULTI \
    --inference_output $INFERENCE_OUTPUT_MULTI \
    --working_dir $BASE_DIR \
    --bert_encoder \
    --bert_full_embeddings \
    --coverage \
    --debias_checkpoint $MODEL_DIR \
    --debias_weight 1.3 \
    --no_tok_enrich \
    --pointer_generator \
    --test_batch_size 1

python ../utils/generate_output.py \
    --in_file $INFERENCE_OUTPUT_SINGLE \
    --out_file $BASE_DIR/output_modular_concurrent_single.txt

python ../utils/generate_output.py \
    --in_file $INFERENCE_OUTPUT_MULTI \
    --out_file $BASE_DIR/output_modular_concurrent_multi.txt

rm $BASE_DIR/input_single.txt
rm $BASE_DIR/input_multi.txt
