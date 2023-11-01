#python scripts/tag_cci_metrics.py \
#    --examples_path outputs/processed_examples/disc_eval_mt-anaphora-marian-small-scat.tsv \
#    --model_name context-mt/scat-marian-small-ctx4-cwd1-en-fr \
#    --model_type marian-small


python scripts/tag_cci_metrics.py \
    --examples_path outputs/processed_examples/disc_eval_mt-lexical-choice-marian-small-scat.tsv \
    --model_name context-mt/scat-marian-small-ctx4-cwd1-en-fr \
    --model_type marian-small


 python scripts/tag_cci_metrics.py \
     --examples_path outputs/processed_examples/disc_eval_mt-anaphora-marian-big-scat.tsv \
     --model_name context-mt/scat-marian-big-ctx4-cwd1-en-fr \
     --model_type marian-big
 
 
 python scripts/tag_cci_metrics.py \
     --examples_path outputs/processed_examples/disc_eval_mt-lexical-choice-marian-big-scat.tsv \
     --model_name context-mt/scat-marian-big-ctx4-cwd1-en-fr \
     --model_type marian-big
 
 
 python scripts/tag_cci_metrics.py \
     --examples_path outputs/processed_examples/disc_eval_mt-anaphora-mbart50-1toM-scat.tsv \
     --model_name context-mt/scat-mbart50-1toM-ctx4-cwd1-en-fr \
     --model_type mbart50-1toM


python scripts/tag_cci_metrics.py \
    --examples_path outputs/processed_examples/disc_eval_mt-lexical-choice-mbart50-1toM-scat.tsv \
    --model_name context-mt/scat-mbart50-1toM-ctx4-cwd1-en-fr \
    --model_type mbart50-1toM

# Target context models

python scripts/tag_cci_metrics.py \
    --examples_path outputs/processed_examples/disc_eval_mt-anaphora-marian-small-scat-target.tsv \
    --model_name context-mt/scat-marian-small-target-ctx4-cwd0-en-fr \
    --model_type marian-small


python scripts/tag_cci_metrics.py \
    --examples_path outputs/processed_examples/disc_eval_mt-lexical-choice-marian-small-scat-target.tsv \
    --model_name context-mt/scat-marian-small-target-ctx4-cwd0-en-fr \
    --model_type marian-small


python scripts/tag_cci_metrics.py \
    --examples_path outputs/processed_examples/disc_eval_mt-anaphora-marian-big-scat-target.tsv \
    --model_name context-mt/scat-marian-big-target-ctx4-cwd0-en-fr \
    --model_type marian-big


python scripts/tag_cci_metrics.py \
    --examples_path outputs/processed_examples/disc_eval_mt-lexical-choice-marian-big-scat-target.tsv \
    --model_name context-mt/scat-marian-big-target-ctx4-cwd0-en-fr \
    --model_type marian-big


python scripts/tag_cci_metrics.py \
    --examples_path outputs/processed_examples/disc_eval_mt-anaphora-mbart50-1toM-scat-target.tsv \
    --model_name context-mt/scat-mbart50-1toM-target-ctx4-cwd0-en-fr \
    --model_type mbart50-1toM


python scripts/tag_cci_metrics.py \
    --examples_path outputs/processed_examples/disc_eval_mt-lexical-choice-mbart50-1toM-scat-target.tsv \
    --model_name context-mt/scat-mbart50-1toM-target-ctx4-cwd0-en-fr \
    --model_type mbart50-1toM