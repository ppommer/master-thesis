###############
MODEL=strap_large
TOP_P=0.6
###############

MODEL_DIR=style_paraphrase/models/$MODEL
INPUT=/home/ppommer/repos/master-thesis/neutralizing-bias/src/inference/concurrent/output_concurrent.txt
OUTPUT=inference/concurrent_strap/output_concurrent_strap.txt

INPUT_SINGLE=datasets/WNC/singleword_biased_test.txt
INPUT_MULTI=datasets/WNC/multiword_biased_test.txt
GOLD_SINGLE=datasets/WNC/singleword_neutral_test.txt
GOLD_MULTI=datasets/WNC/multiword_neutral_test.txt

python strap_many.py \
    --input $INPUT \
    --output $OUTPUT \
    --model_dir $MODEL_DIR \
    --batch_size 1 \
    --top_p_value $TOP_P

python utils/evaluate.py \
    --pred_data $OUTPUT \
    --output inference/concurrent_strap/stats_concurrent_strap_single.txt \
    --gold_data $GOLD_SINGLE \
    --in_data $INPUT_SINGLE

python utils/evaluate.py \
    --pred_data $OUTPUT \
    --output inference/concurrent_strap/stats_concurrent_strap_multi.txt \
    --gold_data $GOLD_MULTI \
    --in_data $INPUT_MULTI
