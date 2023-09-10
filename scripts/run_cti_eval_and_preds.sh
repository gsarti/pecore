for model in marian-small marian-big mbart50-1toM;
do
    
    python scripts/evaluate_tagged_metrics.py \
        --scores_path outputs/scores/scat/scat-${model}-scat-cti.tsv \
        --eval_mode cti \
        --average_example_scores \
        --metrics random pcxmi kl_divergence \
        --save_preds \
        --output_dir outputs/metrics_evals/scat \
        --dataset scat
    
    python scripts/evaluate_tagged_metrics.py \
        --scores_path outputs/scores/scat/scat-${model}-scat-target-cti.tsv \
        --eval_mode cti \
        --average_example_scores \
        --metrics random pcxmi kl_divergence \
        --save_preds \
        --output_dir outputs/metrics_evals/scat \
        --dataset scat
    
    for split in anaphora lexical-choice;
    do
        python scripts/evaluate_tagged_metrics.py \
            --scores_path outputs/scores/disc_eval_mt/${split}/disc_eval_mt-${split}-${model}-scat-cti.tsv \
            --eval_mode cti \
            --average_example_scores \
            --metrics random pcxmi kl_divergence \
            --save_preds \
            --output_dir outputs/metrics_evals/disc_eval_mt/${split} \
            --dataset disc_eval_mt
        
        python scripts/evaluate_tagged_metrics.py \
            --scores_path outputs/scores/disc_eval_mt/${split}/disc_eval_mt-${split}-${model}-scat-target-cti.tsv \
            --eval_mode cti \
            --average_example_scores \
            --metrics random pcxmi kl_divergence \
            --save_preds \
            --output_dir outputs/metrics_evals/disc_eval_mt/${split} \
            --dataset disc_eval_mt
    done
done