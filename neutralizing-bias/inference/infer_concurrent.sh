BASE_DIR=concurrent
MODEL_DIR=../models/concurrent.ckpt

INPUT_SINGLE=../data/WNC/single_test.txt
INPUT_MULTI=../data/WNC/multi_test.txt

OUTPUT_SINGLE=$BASE_DIR/results_concurrent_single.txt
OUTPUT_MULTI=$BASE_DIR/results_concurrent_multi.txt

python ../models/modular.ckptjoint/inference.py \
    --test $INPUT_SINGLE \
    --inference_output $OUTPUT_SINGLE \
    --working_dir $BASE_DIR \
    --bert_encoder \
    --bert_full_embeddings \
    --coverage \
    --debias_checkpoint $MODEL_DIR \
    --debias_weight 1.3 \
    --no_tok_enrich \
    --pointer_generator \
    --test_batch_size 1

python ../models/modular.ckptjoint/inference.py \
    --test $INPUT_MULTI \
    --inference_output $OUTPUT_MULTI \
    --working_dir $BASE_DIR \
    --bert_encoder \
    --bert_full_embeddings \
    --coverage \
    --debias_checkpoint $MODEL_DIR \
    --debias_weight 1.3 \
    --no_tok_enrich \
    --pointer_generator \
    --test_batch_size 1

python ../models/modular.ckptutils/generate_output.py \
    --in_file $OUTPUT_SINGLE \
    --out_file $BASE_DIR/output_concurrent_single.txt

python ../models/modular.ckptutils/generate_output.py \
    --in_file $OUTPUT_MULTI \
    --out_file $BASE_DIR/output_concurrent_multi.txt
