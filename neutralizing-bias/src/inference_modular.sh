BASE_DIR=inference/modular
INPUT=bias_data/WNC/biased.word.test
OUTPUT=$BASE_DIR/results_modular.txt

python joint/inference.py \
    --test $INPUT \
    --inference_output $OUTPUT \
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
    --in_file $OUTPUT \
    --out_file $BASE_DIR/output_modular.txt \
    --html_file $BASE_DIR/output_modular.html
