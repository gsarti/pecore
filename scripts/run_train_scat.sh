#!/bin/bash
#SBATCH --job-name=ctx4-cwd1-scat-marian-big
#SBATCH --time=06:00:00
#SBATCH --mem=30GB
#SBATCH --gpus-per-node=a100.20gb:1
#SBATCH --output=/home3/p305238/slurm_logs/%x.%j.out
 
module purge
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.7.0

cd /home3/p305238
source venv/bin/activate
 
python3 --version
which python3

export HF_HOME=/scratch/p305238/hf_cache

accelerate launch /home3/p305238/scripts/ctx_train/run_translation.py \
    --model_name_or_path context-mt/iwslt17-marian-big-ctx4-cwd1-en-fr \
    --source_lang en_XX \
    --target_lang fr_XX \
    --dataset_name inseq/scat \
    --dataset_config_name sentences \
    --output_dir /scratch/p305238/scat-marian-big-ctx4-cwd1-en-fr \
    --num_beams 5 \
    --max_source_length 512 \
    --max_target_length 128 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 8 \
    --num_warmup_steps 0 \
    --learning_rate 5e-5 \
    --push_to_hub \
    --hub_model_id context-mt/scat-marian-big-ctx4-cwd1-en-fr \
    --hub_token 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 \
    --checkpointing_steps 1000 \
    --logging_steps 200 \
    --with_tracking \
    --report_to tensorboard \
    --context_size 4 \
    --sample_context \
    --context_word_dropout 0.1
 
deactivate