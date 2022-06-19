export DATA_DIR=/home/ppommer/repos/master-thesis/style-transfer-paraphrase/datasets/WNC/WNC_biased_word

BASE_DIR=/home/ppommer/repos/master-thesis/style-transfer-paraphrase/style_paraphrase

python -m torch.distributed.launch --nproc_per_node=1 $BASE_DIR/run_lm_finetuning.py \
    --output_dir=$BASE_DIR/models/WNC_biased_word \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --data_dir=$DATA_DIR \
    --save_steps 500 \
    --logging_steps 20 \
    --save_total_limit -1 \
    --evaluate_during_training \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --per_gpu_train_batch_size 5 \
    --job_id WNC_neutral_large \
    --learning_rate 5e-5 \
    --prefix_input_type paraphrase_250 \
    --global_dense_feature_list none \
    --specific_style_train 0 \
    --optimizer adam
