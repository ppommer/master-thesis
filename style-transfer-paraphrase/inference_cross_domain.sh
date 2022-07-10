MODEL_DIR=style_paraphrase/models/WNC_large

for SET in ibc news speeches; do

    echo strap_large_${SET}
    INPUT=datasets/cross_domain/${SET}_biased.txt
    OUTPUT=inference/cross_domain/output_strap_${SET}.txt

    python strap_many.py \
        --input ${INPUT} \
        --output ${OUTPUT} \
        --model_dir ${MODEL_DIR} \
        --batch_size 1 \
        --top_p_value 0.6

done