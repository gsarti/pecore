for model in marian-small marian-big mbart50-1toM;
do
    for split in anaphora lexical-choice;
    do

        #python scripts/generate_examples.py \
        #    --dataset disc_eval_mt \
        #    --dataset_config ${split} \
        #    --model_name context-mt/scat-${model}-ctx4-cwd1-en-fr \
        #    --src_lang eng \
        #    --tgt_lang fra \
        #    --model_id ${model}-scat \
        #    --model_type ${model} \
        #    --has_context \
        #    --has_contrast
        
        python scripts/generate_examples.py \
            --dataset disc_eval_mt \
            --dataset_config ${split} \
            --model_name context-mt/scat-${model}-target-ctx4-cwd0-en-fr \
            --src_lang eng \
            --tgt_lang fra \
            --model_id ${model}-scat-target \
            --model_type ${model} \
            --has_context \
            --has_contrast \
            --has_target_context
    
    done
done