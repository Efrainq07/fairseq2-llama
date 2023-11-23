from fairseq2.models.sequence import SequenceBatch
from fairseq2.data import VocabularyInfo
from fairseq2.models.llama import create_llama_model, LLaMATokenizer, LLaMAConfig
from torch import argmax
import torch

tokenizer = LLaMATokenizer('./downloads/tokenizer.model')

model_config = LLaMAConfig(
        model_dim=2048,
        max_seq_len=2048,
        vocab_info=VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        ),
        num_layers=22,
        num_attn_heads=32,
        num_key_value_heads=4,
        ffn_inner_dim=1024*8,
        ffn_inner_dim_to_multiple=256,
        dropout_p=0.1,
    )
checkpoint = torch.load('./downloads/TinyLlama-1.1B-intermediate-step-955k-token-2T-fairseq2.bin')

model = create_llama_model(model_config)
model.load_state_dict(checkpoint)

token_encoder = tokenizer.create_encoder()
token_decoder = tokenizer.create_decoder()

tokenized_input = token_encoder('Nathan had a strong first season with the Rangers').reshape(1,-1)

sequences = SequenceBatch(tokenized_input, None)

embedding_output = model(sequences)
output_tokens = argmax( embedding_output.logits,2)

print(token_decoder(output_tokens[-1]))


