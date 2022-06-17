#############
ITERATIONS=10
#############

COUNTER=1
TEST=bias_data/WNC/biased.word.test

for _ in $(seq $ITERATIONS); do
    INFERENCE_OUTPUT=inference/iter_$ITERATIONS/output_modular_$COUNTER.txt

    # echo $COUNTER
    # echo "inference.py INPUT: $TEST"
    # echo "inference.py OUTPUT: $INFERENCE_OUTPUT"

    python joint/inference.py \
        --activation_hidden \
        --bert_full_embeddings \
        --checkpoint models/modular.ckpt \
        --coverage --debias_weight 1.3 \
        --extra_features_top \
        --inference_output $INFERENCE_OUTPUT \
        --pointer_generator \
        --pre_enrich \
        --test $TEST \
        --token_softmax \
        --working_dir inference_modular/

    COUNTER=$(( COUNTER + 1 ))
    TEST=inference/iter_$ITERATIONS/input_modular_$COUNTER.txt

    if [ $COUNTER -le $ITERATIONS ]; then
        # echo "prepare_next.py INPUT: $INFERENCE_OUTPUT"
        # echo "prepare_next.py OUPUT: $TEST"

        python prepare_next.py \
            --input $INFERENCE_OUTPUT.txt
            --output $TEST
    fi
done
