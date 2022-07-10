BASE_DIR=inference/cross_domain

for SET in news speeches; do #ibc news speeches; do
    INPUT=bias_data/cross_domain/${SET}.txt

    echo modular_${SET}

    OUTPUT=${BASE_DIR}/results_modular_${SET}.txt

    python joint/inference.py \
        --test ${INPUT} \
        --inference_output ${OUTPUT} \
        --working_dir ${BASE_DIR} \
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
        --in_file ${OUTPUT} \
        --out_file ${BASE_DIR}/output_modular_${SET}.txt \
        --html_file ${BASE_DIR}/output_modular_${SET}.html

    echo concurrent_${SET}

    OUTPUT=${BASE_DIR}/results_concurrent_${SET}.txt

    python joint/inference.py \
        --test ${INPUT} \
        --inference_output ${OUTPUT} \
        --working_dir ${BASE_DIR} \
        --bert_encoder \
        --bert_full_embeddings \
        --coverage \
        --debias_checkpoint models/concurrent.ckpt \
        --debias_weight 1.3 \
        --no_tok_enrich \
        --pointer_generator \
        --test_batch_size 1

    python utils/generate_output.py \
        --in_file ${OUTPUT} \
        --out_file ${BASE_DIR}/output_concurrent_${SET}.txt

    echo modular_concurrent_${SET}

    python utils/prepare_next.py \
        --input ${BASE_DIR}/results_modular_${SET}.txt \
        --output ${BASE_DIR}/input.txt

    INPUT=${BASE_DIR}/input.txt
    OUTPUT=${BASE_DIR}/results_modular_concurrent_${SET}.txt

    python joint/inference.py \
        --test ${INPUT} \
        --inference_output ${OUTPUT} \
        --working_dir ${BASE_DIR} \
        --bert_encoder \
        --bert_full_embeddings \
        --coverage \
        --debias_checkpoint models/concurrent.ckpt \
        --debias_weight 1.3 \
        --no_tok_enrich \
        --pointer_generator \
        --test_batch_size 1

    python utils/generate_output.py \
        --in_file ${OUTPUT} \
        --out_file ${BASE_DIR}/output_modular_concurrent_${SET}.txt

    echo concurrent_modular_${SET}

    python utils/prepare_next.py \
        --input ${BASE_DIR}/results_concurrent_${SET}.txt \
        --output ${BASE_DIR}/input.txt

    INPUT=${BASE_DIR}/input.txt
    OUTPUT=${BASE_DIR}/results_concurrent_modular_${SET}.txt

    python joint/inference.py \
    --test ${INPUT} \
    --inference_output ${OUTPUT} \
    --working_dir ${BASE_DIR} \
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
        --in_file ${OUTPUT} \
        --out_file ${BASE_DIR}/output_concurrent_modular_${SET}.txt \
        --html_file ${BASE_DIR}/output_concurrent_modular_${SET}.html

done

rm ${BASE_DIR}/input.txt
