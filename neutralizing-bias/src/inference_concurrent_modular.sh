BASE_DIR=inference/concurrent_modular

INPUT_SINGLE=inference/concurrent/results_concurrent_single.txt
TEST_SINGLE=$BASE_DIR/input_single.txt
INFERENCE_OUTPUT_SINGLE=$BASE_DIR/results_concurrent_modular_single.txt
INPUT_MULTI=inference/concurrent/results_concurrent_multi.txt
TEST_MULTI=$BASE_DIR/input_multi.txt
INFERENCE_OUTPUT_MULTI=$BASE_DIR/results_concurrent_modular_multi.txt

python utils/prepare_next.py \
    --input $INPUT_SINGLE \
    --output $TEST_SINGLE

python utils/prepare_next.py \
    --input $INPUT_MULTI \
    --output $TEST_MULTI

python joint/inference.py \
    --test $TEST_SINGLE \
    --inference_output $INFERENCE_OUTPUT_SINGLE \
    --working_dir $BASE_DIR \
    --checkpoint models/modular.ckpt \
    --activation_hidden \
    --bert_full_embeddings \
    --coverage \
    --debias_weight 1.3 \
    --extra_features_top \
    --pointer_generator \
    --pre_enrich \
    --token_softmax \
    --test_batch_size 1

python joint/inference.py \
    --test $TEST_MULTI \
    --inference_output $INFERENCE_OUTPUT_MULTI \
    --working_dir $BASE_DIR \
    --checkpoint models/modular.ckpt \
    --activation_hidden \
    --bert_full_embeddings \
    --coverage \
    --debias_weight 1.3 \
    --extra_features_top \
    --pointer_generator \
    --pre_enrich \
    --token_softmax \
    --test_batch_size 1

python utils/generate_output.py \
    --in_file $INFERENCE_OUTPUT_SINGLE \
    --out_file $BASE_DIR/output_concurrent_modular_single.txt \
    --html_file $BASE_DIR/output_concurrent_modular_single.html

python utils/generate_output.py \
    --in_file $INFERENCE_OUTPUT_MULTI \
    --out_file $BASE_DIR/output_concurrent_modular_multi.txt \
    --html_file $BASE_DIR/output_concurrent_modular_multi.html

rm $BASE_DIR/input_single.txt
rm $BASE_DIR/input_multi.txt
