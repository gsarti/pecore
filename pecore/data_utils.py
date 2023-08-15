import inseq
import stanza

nlp = stanza.Pipeline(lang="en", processors="tokenize", download_method=None)


def get_formatted_examples(
    dataset,
    model_name: str = None,
    force_gen: bool = False,
    has_context: bool = True,
    has_lang_tag: bool = False,
    has_target_context: bool = False,
    start_idx: int = None,
    max_idx: int = None,
) -> List[Dict[str, str]]:
    if max_idx is None:
        max_idx = len(dataset)
    if start_idx is None:
        start_idx = 0
    model = inseq.load_model(model_name, "saliency")
    generate_kwargs = {}
    if has_lang_tag:
        model.tokenizer.src_lang = "en_XX"
        model.tokenizer.tgt_lang = "fr_XX"
        generate_kwargs["forced_bos_token_id"] = model.tokenizer.lang_code_to_id["fr_XX"]
    examples = []
    for idx, ex in tqdm(enumerate(dataset), total=max_idx):
        if idx < start_idx:
            continue
        if max_idx is not None and idx >= max_idx:
            break
        if not force_gen:
            ctx_tgt = None
            if has_context:
                contrast_sources = ex["context_en"] + "<brk> " + ex["en"]
            else:
                contrast_sources = ex["context_en"] + " " + ex["en"]
                ctx_tgt = " ".join(
                    [
                        model.generate(s.text, max_new_tokens=128, **generate_kwargs)[0]
                        for s in nlp(ex["context_en"]).sentences
                    ]
                )
                decoder_input = model.encode(ctx_tgt, as_targets=True).to(model.device)
                generate_kwargs["decoder_input_ids"] = decoder_input.input_ids
                if has_lang_tag:
                    lang_id_tensor = torch.tensor([model.tokenizer.lang_code_to_id["fr_XX"]]).to(model.device)
                    # Prepend the ID tensor to the original tensor along the first dimension (rows)
                    generate_kwargs["decoder_input_ids"] = torch.cat(
                        (lang_id_tensor.unsqueeze(0), generate_kwargs["decoder_input_ids"]), dim=1
                    )
            encoded_sources = model.encode(contrast_sources, as_targets=False).to(model.device)
            generation_out = model.model.generate(
                input_ids=encoded_sources.input_ids,
                attention_mask=encoded_sources.attention_mask,
                return_dict_in_generate=True,
                **generate_kwargs,
            )
            encoded_sources = encoded_sources.to("cpu")
            if not has_context:
                decoder_input = decoder_input.to("cpu")
            ctx_gen = (
                model.tokenizer.batch_decode(
                    generation_out.sequences, skip_special_tokens=False if has_target_context else True
                )[0]
                .replace("<pad>", "")
                .replace("</s>", "")
            )
            del generation_out
            torch.cuda.empty_cache()
            if has_context and has_target_context:
                ctx_tgt = ctx_gen.split("<brk>")[0].strip() + "<brk>"
            start_pos = len(ctx_tgt) if ctx_tgt else 0
            ctx_gen = ctx_gen[start_pos:]
        tgt = ex["fr"] if force_gen else ctx_gen
        ctx_tgt = ex["context_fr"] if force_gen else ctx_tgt
        examples.append(
            {
                "src_en": ex["en"],
                "tgt_fr": tgt,
                "src_en_ctx": contrast_sources,
                "tgt_fr_ctx": ctx_tgt,
                "src_en_with_tags": ex["en_with_tags"],
                "orig_fr": ex["fr"],
                "orig_fr_with_tags": ex["fr_with_tags"],
            }
        )
        if idx < 3:
            print(f"FULL EXAMPLE: {ex}")
            print(f"SRC: {ex['en']}")
            print(f"TGT: {tgt}")
            print(f"SRC CTX: {contrast_sources}")
            print(f"TGT CTX: {ctx_tgt}")
    return examples
