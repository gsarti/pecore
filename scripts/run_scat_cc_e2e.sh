for model in mbart50-1toM;
do
    for metric in kl_divergence pcxmi random;
    do
        python scripts/tag_cci_metrics.py \
            --output_dir outputs/scores/scat \
            --examples_path outputs/processed_examples/scat-${model}-scat.tsv \
            --model_name context-mt/scat-${model}-ctx4-cwd1-en-fr \
            --model_type ${model} \
            --target_tags_path outputs/metrics_evals/scat/scat-${model}-scat-cti-avg-${metric}-preds.txt

        python scripts/tag_cci_metrics.py \
            --output_dir outputs/scores/scat \
            --examples_path outputs/processed_examples/scat-${model}-scat-target.tsv \
            --model_name context-mt/scat-${model}-target-ctx4-cwd0-en-fr \
            --model_type ${model} \
            --target_tags_path outputs/metrics_evals/scat/scat-${model}-scat-target-cti-avg-${metric}-preds.txt
    done
done