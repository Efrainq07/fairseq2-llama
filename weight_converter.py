import torch
from fairseq2.models.utils.checkpoint import convert_model_state_dict


key_map = {
        # fmt: off
        r"^model.layers\.([0-9]+)\.input_layernorm\.":   r"decoder.layers.\1.self_attn_layer_norm.",
        r"^model.layers\.([0-9]+)\.self_attn\.q_proj\.":    r"decoder.layers.\1.self_attn.q_proj.",
        r"^model.layers\.([0-9]+)\.self_attn\.k_proj\.":    r"decoder.layers.\1.self_attn.k_proj.",
        r"^model.layers\.([0-9]+)\.self_attn\.v_proj\.":    r"decoder.layers.\1.self_attn.v_proj.",
        r"^model.layers\.([0-9]+)\.self_attn\.o_proj\.":    r"decoder.layers.\1.self_attn.output_proj.",
        r"^model.layers\.([0-9]+)\.post_attention_layernorm\.":         r"decoder.layers.\1.ffn_layer_norm.",
        r"^model.layers\.([0-9]+)\.mlp\.gate_proj\.": r"decoder.layers.\1.ffn.gate_proj.",
        r"^model.layers\.([0-9]+)\.mlp\.down_proj\.": r"decoder.layers.\1.ffn.output_proj.",
        r"^model.layers\.([0-9]+)\.mlp\.up_proj\.": r"decoder.layers.\1.ffn.inner_proj.",
        r"^model.norm\.":                               r"decoder.layer_norm.",
        r"^model.embed_tokens\.":                     r"decoder_frontend.embed.",
        r"^lm_head\.":                             r"final_proj.",
        # fmt: on
    }

old_checkpoint = torch.load('./downloads/TinyLlama-1.1B-intermediate-step-955k-token-2T.bin')
converted_checkpoint = convert_model_state_dict(old_checkpoint, key_map)


for key in converted_checkpoint:
    print(key, list(converted_checkpoint[key].shape))

torch.save(converted_checkpoint,'./downloads/TinyLlama-1.1B-intermediate-step-955k-token-2T-fairseq2.bin')