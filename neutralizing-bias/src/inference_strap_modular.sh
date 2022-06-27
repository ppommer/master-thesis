DATA=/home/ppommer/repos/master-thesis/style-transfer-paraphrase/inference/strap_large/output_strap_large_6.txt
BASE_DIR=inference/strap_modular

INPUT=$BASE_DIR/input.txt
OUTPUT=$BASE_DIR/results_strap_modular.txt

python utils/prepare_strap.py \
    --input $DATA \
    --output $INPUT \
    --gold bias_data/WNC/test.txt

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
    --out_file $BASE_DIR/output_strap_modular.txt
    --html_file $BASE_DIR/output_strap_modular.html

rm $BASE_DIR/input.txt
