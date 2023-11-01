for model in marian-big;
do
    #python scripts/evaluate_tagged_metrics.py \
    #    --scores_path outputs/scores/scat/scat-${model}-scat-cci.tsv \
    #    --eval_mode cci \
    #    --average_example_scores \
    #    --output_dir outputs/metrics_evals/scat \
    #    --dataset scat \
    #    --example_target_column is_supporting_context \
    #    --model_type ${model}
    #
    #python scripts/evaluate_tagged_metrics.py \
    #    --scores_path outputs/scores/scat/scat-${model}-scat-target-cci.tsv \
    #    --eval_mode cci \
    #    --average_example_scores \
    #    --output_dir outputs/metrics_evals/scat \
    #    --dataset scat \
    #    --example_target_column is_supporting_context \
    #    --model_type ${model} \
    #    --has_target_context

    for metric in kl_divergence pcxmi random;
    do
        python scripts/evaluate_tagged_metrics.py \
            --scores_path outputs/scores/scat/e2e/scat-${model}-scat-tags_${metric}-cci.tsv \
            --eval_mode cci \
            --average_example_scores \
            --output_dir outputs/metrics_evals/scat \
            --dataset scat \
            --example_target_column is_supporting_context \
            --model_type ${model}
        
        python scripts/evaluate_tagged_metrics.py \
            --scores_path outputs/scores/scat/e2e/scat-${model}-scat-target-tags_${metric}-cci.tsv \
            --eval_mode cci \
            --average_example_scores \
            --output_dir outputs/metrics_evals/scat \
            --dataset scat \
            --example_target_column is_supporting_context \
            --model_type ${model} \
            --has_target_context
    done
    
    for split in anaphora lexical-choice;
    do
        #python scripts/evaluate_tagged_metrics.py \
        #    --scores_path outputs/scores/disc_eval_mt/${split}/disc_eval_mt-${split}-${model}-scat-cci.tsv \
        #    --eval_mode cci \
        #    --average_example_scores \
        #    --output_dir outputs/metrics_evals/disc_eval_mt/${split} \
        #    --dataset disc_eval_mt \
        #    --example_target_column is_supporting_context \
        #    --model_type ${model}
        #
        #python scripts/evaluate_tagged_metrics.py \
        #    --scores_path outputs/scores/disc_eval_mt/${split}/disc_eval_mt-${split}-${model}-scat-target-cci.tsv \
        #    --eval_mode cci \
        #    --average_example_scores \
        #    --output_dir outputs/metrics_evals/disc_eval_mt/${split} \
        #    --dataset disc_eval_mt \
        #    --example_target_column is_supporting_context \
        #    --model_type ${model} \
        #    --has_target_context

        for metric in kl_divergence pcxmi random;
        do
            python scripts/evaluate_tagged_metrics.py \
                --scores_path outputs/scores/disc_eval_mt/${split}/e2e/disc_eval_mt-${split}-${model}-scat-tags_${metric}-cci.tsv \
                --eval_mode cci \
                --average_example_scores \
                --output_dir outputs/metrics_evals/disc_eval_mt/${split} \
                --dataset disc_eval_mt \
                --example_target_column is_supporting_context \
                --model_type ${model}
            
            python scripts/evaluate_tagged_metrics.py \
                --scores_path outputs/scores/disc_eval_mt/${split}/e2e/disc_eval_mt-${split}-${model}-scat-target-tags_${metric}-cci.tsv \
                --eval_mode cci \
                --average_example_scores \
                --output_dir outputs/metrics_evals/disc_eval_mt/${split} \
                --dataset disc_eval_mt \
                --example_target_column is_supporting_context \
                --model_type ${model} \
                --has_target_context
        done
    done
done