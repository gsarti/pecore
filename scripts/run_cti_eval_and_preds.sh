for model in marian-small marian-big mbart50-1toM;
do
    
    python scripts/evaluate_tagged_metrics.py \
        --scores_path outputs/scores/scat-${model}-scat-cti.tsv \
        --eval_mode cti \
        --average_example_scores \
        --metrics random pcxmi kl_divergence \
        --save_preds \
        --output_dir outputs/metrics_evals/scat
    
    python scripts/evaluate_tagged_metrics.py \
        --scores_path outputs/scores/scat-${model}-scat-target-cti.tsv \
        --eval_mode cti \
        --average_example_scores \
        --metrics random pcxmi kl_divergence \
        --save_preds \
        --output_dir outputs/metrics_evals/scat
    
    for split in anaphora lexical-choice;
    do
        python scripts/evaluate_tagged_metrics.py \
            --scores_path outputs/scores/disc_eval_mt-${split}-${model}-scat-cti.tsv \
            --eval_mode cti \
            --average_example_scores \
            --metrics random pcxmi kl_divergence \
            --save_preds \
            --output_dir outputs/metrics_evals/disc_eval_mt/${split}
        
        python scripts/evaluate_tagged_metrics.py \
            --scores_path outputs/scores/disc_eval_mt-${split}-${model}-scat-target-cti.tsv \
            --eval_mode cti \
            --average_example_scores \
            --metrics random pcxmi kl_divergence \
            --save_preds \
            --output_dir outputs/metrics_evals/disc_eval_mt/${split}
    done
done