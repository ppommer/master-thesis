BASE_DIR=inference/modular

INPUT_SINGLE=bias_data/WNC_edit/single_test.txt
OUTPUT_SINGLE=$BASE_DIR/results_modular_single.txt
INPUT_MULTI=bias_data/WNC_edit/multi_test.txt
OUTPUT_MULTI=$BASE_DIR/results_modular_multi.txt

python joint/inference.py \
    --test $INPUT_SINGLE \
    --inference_output $OUTPUT_SINGLE \
    --working_dir $BASE_DIR \
    --activation_hidden \
    --bert_full_embeddings \
    --checkpoint models/modular.ckpt \
    --coverage --debias_weight 1.3 \
    --extra_features_top \
    --pointer_generator \
    --pre_enrich \
    --token_softmax \
    --test_batch_size 1

python joint/inference.py \
    --test $INPUT_MULTI \
    --inference_output $OUTPUT_MULTI \
    --working_dir $BASE_DIR \
    --activation_hidden \
    --bert_full_embeddings \
    --checkpoint models/modular.ckpt \
    --coverage --debias_weight 1.3 \
    --extra_features_top \
    --pointer_generator \
    --pre_enrich \
    --token_softmax \
    --test_batch_size 1

python utils/generate_output.py \
    --in_file $OUTPUT_SINGLE \
    --out_file $BASE_DIR/output_modular_single.txt \
    --html_file $BASE_DIR/output_modular_single.html

python utils/generate_output.py \
    --in_file $OUTPUT_MULTI \
    --out_file $BASE_DIR/output_modular_multi.txt \
    --html_file $BASE_DIR/output_modular_multi.html
