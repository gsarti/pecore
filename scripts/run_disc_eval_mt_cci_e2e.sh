for model in marian-big;
do
    for split in anaphora lexical-choice;
    do
        for metric in random;
        do
            python scripts/tag_cci_metrics.py \
                --output_dir outputs/scores/disc_eval_mt/${split} \
                --examples_path outputs/processed_examples/disc_eval_mt-${split}-${model}-scat.tsv \
                --model_name context-mt/scat-${model}-ctx4-cwd1-en-fr \
                --model_type ${model} \
                --target_tags_path outputs/metrics_evals/disc_eval_mt/${split}/disc_eval_mt-${split}-${model}-scat-cti-avg-${metric}-preds.txt
        done
    done
done

for model in marian-small;
do
    for split in anaphora lexical-choice;
    do
        for metric in random;
        do
        
            if [ ${split} == "lexical-choice" ]; then
                python scripts/tag_cci_metrics.py \
                --output_dir outputs/scores/disc_eval_mt/${split} \
                --examples_path outputs/processed_examples/disc_eval_mt-${split}-${model}-scat.tsv \
                --model_name context-mt/scat-${model}-ctx4-cwd1-en-fr \
                --model_type ${model} \
                --target_tags_path outputs/metrics_evals/disc_eval_mt/${split}/disc_eval_mt-${split}-${model}-scat-cti-avg-${metric}-preds.txt
            fi
        
            python scripts/tag_cci_metrics.py \
                --output_dir outputs/scores/disc_eval_mt/${split} \
                --examples_path outputs/processed_examples/disc_eval_mt-${split}-${model}-scat-target.tsv \
                --model_name context-mt/scat-${model}-target-ctx4-cwd0-en-fr \
                --model_type ${model} \
                --target_tags_path outputs/metrics_evals/disc_eval_mt/${split}/disc_eval_mt-${split}-${model}-scat-target-cti-avg-${metric}-preds.txt
        done
    done
done