# PECoRe

## Translate with a Context-Aware NMT Model

```shell
python scripts/translate.py \
    --model_type mbart50-1toM \
    --model_id mbart50-1toM-scat \
    --model_name context-mt/scat-mbart50-1toM-ctx4-cwd1-en-fr \
    --context_size 4 \ 
    --dataset scat \
    --context_word_dropout 1
```

## Train a Context-Aware NMT Model

Context-aware NMT models are trained using the `train.py` script. The script is a modification of the original
[`run_translation_no_trainer.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation_no_trainer.py). The script adds the following fields for contextual model training:

- `context_size`: The number of context sentences to use for training. The default value is 0 (sentence-level training).

- `sample_context`: If set, the size of the context for every example is sampled from a uniform distribution between 0 and `context_size` (inclusive). If not passed and `context_size` is greater than 0, the context size is always equal to `context_size`.

- `context_word_dropout`: Probability between 0 and 1 of dropping a word from the context. The default value is 0 (no dropout).

- `use_target_context`: If set, the context is also included in the translated text for the training loss. In that case, the output format for an input `src_ctx <brk> src` becomes `tgt_ctx <brk> tgt`. Otherwise the output format is `tgt` (only `src` is translated).

<details>
    <summary>Example usage</summary>


Here is an example of fine-tuning an mBART 1-to-50 model on the context-augmented IWSLT17 dataset with up to 4 context sentences and a 10% context word dropout:

```shell
accelerate launch scripts/train.py \
    --model_name_or_path facebook/mbart-large-50-one-to-many-mmt \
    --source_lang en_XX \
    --target_lang fr_XX \
    --dataset_name gsarti/iwslt2017_context \
    --dataset_config_name iwslt2017-en-fr \
    --output_dir outputs/models/iwslt17-mbart50-1toM-ctx4-cwd1-en-fr \
    --num_beams 5 \
    --max_source_length 512 \
    --max_target_length 128 \
    --num_train_epochs 20 \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 8 \
    --num_warmup_steps 500 \
    --learning_rate 3e-4 \
    --checkpointing_steps epoch \
    --with_tracking \
    --report_to tensorboard \
    --context_size 4 \
    --sample_context \
    --context_word_dropout 0.1 
```

Here is an example of continuing the fine-tuning of a context-aware En->Fr OpusMT model on the training portion of SCAT with up to 4 context sentences and a 10% context word dropout:

```shell
accelerate launch scripts/train.py \
    --model_name_or_path context-mt/iwslt17-marian-big-ctx4-cwd1-en-fr \
    --dataset_name inseq/scat \
    --dataset_config_name sentences \
    --output_dir outputs/models/scat-marian-big-ctx4-cwd1-en-fr \
    --num_beams 5 \
    --max_source_length 512 \
    --max_target_length 128 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 8 \
    --num_warmup_steps 0 \
    --learning_rate 5e-5 \
    --checkpointing_steps 1000 \
    --logging_steps 200 \
    --with_tracking \
    --report_to tensorboard \
    --context_size 4 \
    --sample_context \
    --context_word_dropout 0.1
```
</details>
