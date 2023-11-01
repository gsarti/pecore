for model in marian-small marian-big mbart50-1toM;
do
    
    python scripts/evaluate_tagged_metrics.py \
        --scores_path outputs/scores/scat/scat-${model}-scat-cti.tsv \
        --eval_mode cti \
        --metrics random ctx_saliency likelihood_ratio pcxmi kl_divergence \
        --output_dir outputs/metrics_evals/scat \
        --dataset scat \
        --model_type ${model} \
        --average_example_scores \
        --save_per_example_scores 
        #--save_preds \
    
    python scripts/evaluate_tagged_metrics.py \
        --scores_path outputs/scores/scat/scat-${model}-scat-target-cti.tsv \
        --eval_mode cti \
        --metrics random ctx_saliency likelihood_ratio pcxmi kl_divergence \
        --output_dir outputs/metrics_evals/scat \
        --dataset scat \
        --model_type ${model} \
        --average_example_scores \
        --save_per_example_scores \
        #--save_preds \
    
    for split in anaphora lexical-choice;
    do
        python scripts/evaluate_tagged_metrics.py \
            --scores_path outputs/scores/disc_eval_mt/${split}/disc_eval_mt-${split}-${model}-scat-cti.tsv \
            --eval_mode cti \
            --average_example_scores \
            --metrics random ctx_saliency likelihood_ratio pcxmi kl_divergence \
            --output_dir outputs/metrics_evals/disc_eval_mt/${split} \
            --dataset disc_eval_mt \
            --model_type ${model} \
            --save_per_example_scores
            #--save_preds \
        
        python scripts/evaluate_tagged_metrics.py \
            --scores_path outputs/scores/disc_eval_mt/${split}/disc_eval_mt-${split}-${model}-scat-target-cti.tsv \
            --eval_mode cti \
            --average_example_scores \
            --metrics random ctx_saliency likelihood_ratio pcxmi kl_divergence \
            --output_dir outputs/metrics_evals/disc_eval_mt/${split} \
            --dataset disc_eval_mt \
            --model_type ${model} \
            --save_per_example_scores \
            #--save_preds \
    done
done