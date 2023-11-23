import logging
import sys

import torch

from fairseq2.nn.lora import wrap_lora
from fairseq2.data import VocabularyInfo
from fairseq2.models.llama import create_llama_model, get_llama_lora_config, LLaMATokenizer, LLaMAConfig
from llama_trainer import (
    LLaMAFinetuneParams,
    LLaMAFinetune,
    CleanWikiText103,
    BatchingConfig,
    LLaMADataLoader)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logger = logging.getLogger(__name__)

#Create LLaMA dataloader
logger.info('Creating LLaMA Tokenizer...')
tokenizer = LLaMATokenizer('./downloads/tokenizer.model')

logger.info('Creating WikiText103 datapipes...')
train_datapipe = CleanWikiText103(split='train')
valid_datapipe = CleanWikiText103(split='valid')

logger.info('Creating batching confguration...')
batching_config = BatchingConfig(
                    batch_size = 3,
                    float_dtype = torch.float32)

logger.info('Creating data loaders...')
train_dataloader = LLaMADataLoader(tokenizer,batching_config,train_datapipe)

#Create finetuning parameters
logger.info('Creating finetuning parameters...')
finetune_params = LLaMAFinetuneParams(save_model_path='.')

#Create LLaMA model instance
logger.info('Creating LLaMA model instance...')
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
llama_model = create_llama_model(model_config,device=torch.device("cpu"))

checkpoint = torch.load('./downloads/TinyLlama-1.1B-intermediate-step-955k-token-2T-fairseq2.bin')

llama_model.load_state_dict(checkpoint)

lora_config = get_llama_lora_config()
model = wrap_lora(llama_model,lora_config).to('cuda')

logger.info('Creating finetuning instance...')
finetuning = LLaMAFinetune(
                    model,
                    finetune_params,
                    train_dataloader,
                    None
                    )

logger.info('Starting training...')
#Run finetuning
finetuning.run()
