for model in marian-small marian-big mbart50-1toM;
do
    
    python scripts/evaluate_tagged_metrics.py \
        --scores_path outputs/scores/scat/scat-${model}-scat-cci.tsv \
        --eval_mode cci \
        --average_example_scores \
        --metrics random saliency_contrast_prob_diff saliency_kl_divergence attention_default \
        --output_dir outputs/metrics_evals/scat \
        --dataset scat \
        --model_type ${model} \
        --example_target_column is_supporting_context \
        --save_per_example_scores 
    
    python scripts/evaluate_tagged_metrics.py \
        --scores_path outputs/scores/scat/scat-${model}-scat-target-cci.tsv \
        --eval_mode cci \
        --metrics random saliency_contrast_prob_diff saliency_kl_divergence attention_default \
        --output_dir outputs/metrics_evals/scat \
        --dataset scat \
        --model_type ${model} \
        --average_example_scores \
        --example_target_column is_supporting_context \
        --save_per_example_scores 

    for metric in kl_divergence;
    do

        python scripts/evaluate_tagged_metrics.py \
            --scores_path outputs/scores/scat/e2e/scat-${model}-scat-tags_${metric}-cci.tsv \
            --eval_mode cci \
            --average_example_scores \
            --metrics random saliency_contrast_prob_diff saliency_kl_divergence attention_default \
            --output_dir outputs/metrics_evals/scat \
            --dataset scat \
            --model_type ${model} \
            --example_target_column is_supporting_context \
            --save_per_example_scores 

        python scripts/evaluate_tagged_metrics.py \
            --scores_path outputs/scores/scat/e2e/scat-${model}-scat-target-tags_${metric}-cci.tsv \
            --eval_mode cci \
            --metrics random saliency_contrast_prob_diff saliency_kl_divergence attention_default \
            --output_dir outputs/metrics_evals/scat \
            --dataset scat \
            --model_type ${model} \
            --average_example_scores \
            --example_target_column is_supporting_context \
            --save_per_example_scores
    done
    
    for split in anaphora lexical-choice;
    do

        python scripts/evaluate_tagged_metrics.py \
            --scores_path outputs/scores/disc_eval_mt/${split}/disc_eval_mt-${split}-${model}-scat-cci.tsv \
            --eval_mode cci \
            --average_example_scores \
            --metrics random saliency_contrast_prob_diff saliency_kl_divergence attention_default \
            --output_dir outputs/metrics_evals/disc_eval_mt/${split} \
            --dataset disc_eval_mt \
            --model_type ${model} \
            --example_target_column is_supporting_context \
            --save_per_example_scores 
        
        python scripts/evaluate_tagged_metrics.py \
            --scores_path outputs/scores/disc_eval_mt/${split}/disc_eval_mt-${split}-${model}-scat-target-cci.tsv \
            --eval_mode cci \
            --average_example_scores \
            --metrics random saliency_contrast_prob_diff saliency_kl_divergence attention_default \
            --output_dir outputs/metrics_evals/disc_eval_mt/${split} \
            --dataset disc_eval_mt \
            --model_type ${model} \
            --example_target_column is_supporting_context \
            --save_per_example_scores
        
        for metric in kl_divergence;
        do

            python scripts/evaluate_tagged_metrics.py \
                --scores_path outputs/scores/disc_eval_mt/${split}/e2e/disc_eval_mt-${split}-${model}-scat-tags_${metric}-cci.tsv \
                --eval_mode cci \
                --average_example_scores \
                --metrics random saliency_contrast_prob_diff saliency_kl_divergence attention_default \
                --output_dir outputs/metrics_evals/disc_eval_mt/${split} \
                --dataset disc_eval_mt \
                --model_type ${model} \
                --example_target_column is_supporting_context \
                --save_per_example_scores 

            python scripts/evaluate_tagged_metrics.py \
                --scores_path outputs/scores/disc_eval_mt/${split}/e2e/disc_eval_mt-${split}-${model}-scat-target-tags_${metric}-cci.tsv \
                --eval_mode cci \
                --metrics random saliency_contrast_prob_diff saliency_kl_divergence attention_default \
                --output_dir outputs/metrics_evals/disc_eval_mt/${split} \
                --dataset disc_eval_mt \
                --model_type ${model} \
                --average_example_scores \
                --example_target_column is_supporting_context \
                --save_per_example_scores
        done
    done
done