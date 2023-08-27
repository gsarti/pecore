# PECoRe

### Train a Context-Aware NMT Model

Context-aware NMT models are trained using the `train_context_aware_mt_model.py` script. The script is a modification of the original
[`run_translation_no_trainer.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation_no_trainer.py). The script adds the following fields for contextual model training:

- `context_size`: The number of context sentences to use for training. The default value is 0 (sentence-level training).

- `sample_context`: If set, the size of the context for every example is sampled from a uniform distribution between 0 and `context_size` (inclusive). If not passed and `context_size` is greater than 0, the context size is always equal to `context_size`.

- `context_word_dropout`: Probability between 0 and 1 of dropping a word from the context. The default value is 0 (no dropout).

- `use_target_context`: If set, the context is also included in the translated text for the training loss. In that case, the output format for an input `src_ctx <brk> src` becomes `tgt_ctx <brk> tgt`. Otherwise the output format is `tgt` (only `src` is translated).

<details>
    <summary>Example usage</summary>


Here is an example of fine-tuning an mBART 1-to-50 model on the context-augmented IWSLT17 dataset with up to 4 context sentences and a 10% context word dropout:

```shell
accelerate launch scripts/train_context_aware_mt_model.py \
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
accelerate launch scripts/train_context_aware_mt_model.py \
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

## Using the PECoRe CLI

The PECoRe CLI is a command-line interface for running the PECoRe steps on a given model and dataset. The CLI is implemented in the `pecore/cli.py` script and can be used as `pecore-viz` upon installing the package with `pip install -e .`. The current implementation supports the identification of context-sensitive targets (CTI) and the imputation of contextual cues (CCI) for all encoder-decoder models supported by the [Inseq](https://github.com/inseq-team/inseq) framework, including models with language prefix tags (mBART-50, NLLB, M2M100) and models trained with special context tags (e.g. the collection of models found in the [context-mt](https://huggingface.co/context-mt) organization on the HF Hub). The CLI can be used to run the PECoRe steps on a given model and example as follows:

```shell
pecore-viz \
    --model_name context-mt/scat-marian-small-ctx4-cwd1-en-fr \
    --attributions_aggregate_fns sum \
    --model_use_ctx_break \
    --impute_with_contextless_output \
    --force_context_aware_output_prefix \
    --input "Did I mention we stole a cow? A beautiful animal, truly. We brought it to the stable and kept it there for ages.<brk> Sadly, we could not foresee it would disappear."
```

The example above produces the following output, correctly highlighting the dependence on the pronoun "il" on the nouns "cow" and "animal" in the context.

```shell
Context with contextual cues (std λ=1.00) followed by output sentence
with context-sensitive target spans (std λ=1.00):

Input context:  Did I mention we stole a cow? A beautiful animal, truly. We brought it to the stable and kept it there for ages.
Input current:  Sadly, we could not foresee it would disappear.
Context-aware output:   Malheureusement, nous n'avons pas pu prévoir qu'il disparaîtrait.
Using '<brk> ' to separate context and current inputs.

#1. (CTI |kl_divergence| > 0.14, CCI |saliency| > 0.71)
Contextless output:     Malheureusement, nous n'avons pas pu prévoir qu'il disparaîtrait.
Current output:  Malheureusement, nous n'avons pas pu prévoir qu'il(0.412) disparaîtrait.
Input context:   Did I mention we stole a cow(1.524)? A beautiful animal(1.472), truly. We brought it to the stable and kept it 
there for ages.
```

When using the CLI to run a regular model, an additional step will be needed to specify the position of the context break in model's generation if an output is not forced by the user. Here is an example using the regular mBART-50 model from the HF Hub:

```shell
pecore-viz \
    --model_name facebook/mbart-large-50-one-to-many-mmt \
    --input_lang eng --output_lang fra --model_type mbart50-1toM \
    --impute_with_contextless_output \
    --force_context_aware_output_prefix \
    --input "Did I mention we stole a cow? A beautiful animal, truly. We brought it to the stable and kept it there for ages.<brk> Sadly, we could not foresee it would disappear."
```

The user will be prompted with the following message:

```shell
The following output was generate by the model: J’ai mentionné que nous avons volé une vache, c’est vraiment un beau animal, que nous avons emmené à l’élevage et que nous l’avons gardée pendant des époques. Malheureusement, nous n’avons pas pu prévoir qu’elle disparaîtrait.
Rewrite it here by adding '<brk> ' wherever appropriate to mark context break:
```

The user can then rewrite the output by adding `<brk> ` wherever appropriate to mark the context break:

```shell
J’ai mentionné que nous avons volé une vache, c’est vraiment un beau animal, que nous avons emmené à l’élevage et que nous l’avons gardée pendant des époques.<brk> Malheureusement, nous n’avons pas pu prévoir qu’elle disparaîtrait.
```

The final output will be:

```shell
Context with contextual cues (std λ=1.00) followed by output sentence
with context-sensitive target spans (std λ=1.00):

Input context:  Did I mention we stole a cow? A beautiful animal, truly. We brought it to the stable and kept it there for ages.
Input current:  Sadly, we could not foresee it would disappear.
Output context: J’ai mentionné que nous avons volé une vache, c’est vraiment un beau animal, que nous avons emmené à l’élevage et que nous l’avons gardée pendant 
des époques.
Context-aware output:   J’ai mentionné que nous avons volé une vache, c’est vraiment un beau animal, que nous avons emmené à l’élevage et que nous l’avons gardée 
pendant des époques. Malheureusement, nous n’avons pas pu prévoir qu’elle disparaîtrait.
Using language tags for model type 'mbart50-1toM' (eng -> fra).

#1. (CTI |kl_divergence| > 1.08, CCI |saliency| > 0.00)
Contextless output:     Malheureusement, nous n'avons pas pu prévoir sa disparition.
Current output:  Malheureusement, nous n’(3.505)avons pas pu prévoir qu’elle disparaîtrait.
Input context:   Did I mention we stole a cow? A beautiful animal, truly. We brought it to the stable(0.002) and kept it there for ages.
Output context:  J’(0.004)ai mentionné que nous avons volé une vache, c’(0.002)est vraiment un beau animal, que nous avons emmené à l’(0.003)élevage et que nous 
l’(0.007)avons gardée pendant des époques.
```

In this case, we see the model opts to generate the curved apostrophe `’` rather than the straight one `'` used by default in the contextless output to stick to the output context style, employing that character on several occasions (identified as contextual cues by PECoRe).

## Reproducing the Paper Results

### Translate with a Context-Aware NMT Model

```shell
python scripts/translate.py \
    --model_type mbart50-1toM \
    --model_id mbart50-1toM-scat \
    --model_name context-mt/scat-mbart50-1toM-ctx4-cwd1-en-fr \
    --context_size 4 \ 
    --dataset scat \
    --context_word_dropout 1
```

### Evaluate a Context-Aware NMT Model

```shell
python scripts/evaluate_mt_outputs.py \
    --filepath outputs/translations/ctx/scat-mbart50-1toM-scat.txt \
    --model_id mbart50-1toM-scat \
    --dataset scat \
    --src_lang eng \
    --tgt_lang fra \
    --metrics bleu comet accuracy
```

### Create examples for running PECoRe steps

```shell
python scripts/generate_examples.py \
    --dataset scat \
    --model_name context-mt/scat-marian-small-ctx4-cwd1-en-fr \
    --src_lang eng \
    --tgt_lang fra \
    --model_id marian-small-scat \
    --model_type marian-small \
    --has_context \
    --has_contrast

python scripts/generate_examples.py \
    --dataset scat \
    --model_name context-mt/scat-mbart50-1toM-target-ctx4-cwd0-en-fr \
    --src_lang eng \
    --tgt_lang fra \
    --model_id mbart50-1toM-scat-target \
    --model_type mbart50-1toM \
    --has_context \
    --has_target_context \
    --has_contrast
```

### PECoRe Step 1: Context-sensitive Target Identification (CTI)

```shell
python scripts/tag_cti_metrics.py \
    --examples_path outputs/processed_examples/scat-marian-small-scat.tsv \
    --model_name context-mt/scat-marian-small-ctx4-cwd1-en-fr \
    --model_type marian-small

python scripts/tag_cti_metrics.py \
    --examples_path outputs/processed_examples/scat-marian-big-scat.tsv \
    --model_name context-mt/scat-marian-big-ctx4-cwd1-en-fr \
    --model_type marian-big

python scripts/tag_cti_metrics.py \
    --examples_path outputs/processed_examples/scat-mbart50-1toM-scat.tsv \
    --model_name context-mt/scat-mbart50-1toM-ctx4-cwd1-en-fr \
    --model_type mbart50-1toM

python scripts/tag_cti_metrics.py \
    --examples_path outputs/processed_examples/scat-marian-small-scat-target.tsv \
    --model_name context-mt/scat-marian-small-target-ctx4-cwd0-en-fr \
    --model_type marian-small

python scripts/tag_cti_metrics.py \
    --examples_path outputs/processed_examples/scat-marian-big-scat-target.tsv \
    --model_name context-mt/scat-marian-big-target-ctx4-cwd0-en-fr \
    --model_type marian-big

python scripts/tag_cti_metrics.py \
    --examples_path outputs/processed_examples/scat-mbart50-1toM-scat-target.tsv \
    --model_name context-mt/scat-mbart50-1toM-target-ctx4-cwd0-en-fr \
    --model_type mbart50-1toM \
    --start_idx 293
```

### PECoRe Step 2: Contextual Cues Imputation (CCI)

```shell
python scripts/tag_cci_metrics.py \
    --examples_path outputs/processed_examples/scat-marian-small-scat.tsv \
    --model_name context-mt/scat-marian-small-ctx4-cwd1-en-fr \
    --model_type marian-small

python scripts/tag_cci_metrics.py \
    --examples_path outputs/processed_examples/scat-mbart50-1toM-scat-target.tsv \
    --model_name context-mt/scat-mbart50-1toM-target-ctx4-cwd0-en-fr \
    --model_type mbart50-1toM
```

### Evaluate PECoRe Metrics

```shell
python scripts/evaluate_tagged_metrics.py \
    --scores_path outputs/scores/scat-marian-small-scat-cti.tsv \
    --eval_mode cti

python scripts/evaluate_tagged_metrics.py \
    --scores_path outputs/scores/scat-marian-small-scat-cci.tsv \
    --eval_mode cci \
    --example_target_column is_supporting_context
```