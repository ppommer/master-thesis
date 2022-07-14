MODEL_DIR=style_paraphrase/models/strap_

for MODEL in word full large; do
    for SET in ibc news speeches; do

        echo strap_${MODEL}_${SET}
        INPUT=datasets/cross_domain/${SET}_biased.txt
        OUTPUT=inference/cross_domain/output_${MODEL}_${SET}.txt

        python strap_many.py \
            --input ${INPUT} \
            --output ${OUTPUT} \
            --model_dir ${MODEL_DIR}${MODEL} \
            --batch_size 1 \
            --top_p_value 0.0

    done
done