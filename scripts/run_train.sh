#!/bin/bash
#SBATCH --job-name=run_translation_train
#SBATCH --time=10:00:00
#SBATCH --mem=100GB
#SBATCH --gpus-per-node=a100:2
#SBATCH --output=/home3/p305238/slurm_logs/%x.%j.out
 
module purge
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.7.0

cd /home3/p305238
source venv/bin/activate
 
python3 --version
which python3

export HF_HOME=/scratch/p305238/hf_cache

accelerate launch /home3/p305238/scripts/custom_run_translation_no_trainer.py \
    --model_name_or_path facebook/mbart-large-50-one-to-many-mmt \
    --source_lang en_XX \
    --target_lang fr_XX \
    --dataset_name gsarti/iwslt2017_context \
    --dataset_config_name iwslt2017-en-fr \
    --output_dir /scratch/p305238/iwslt17-mbart-ctx-en-fr \
    --num_beams 5 \
    --max_source_length 512 \
    --max_target_length 128 \
    --num_train_epochs 20 \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 8 \
    --num_warmup_steps 500 \
    --learning_rate 3e-4 \
    --push_to_hub \
    --hub_model_id gsarti/iwslt17-mbart-ctx-en-fr \
    --hub_token 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 \
    --checkpointing_steps epoch \
    --with_tracking \
    --report_to tensorboard \
    --context_size 4 \
    --sample_context \
    --context_word_dropout 0.1 
 
deactivate