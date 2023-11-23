from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import pad as pad_tensor
from torchdata.datapipes.iter import IterDataPipe
from fairseq2.models.sequence import SequenceBatch

from fairseq2.models.sequence import SequenceBatch

from fairseq2.models.llama import LLaMATokenizer

@dataclass
class BatchingConfig:
    batch_size: int = 3

    float_dtype: torch.dtype = torch.float16
    """Select between fp16/fp32 for float tensors """

@dataclass
class LLaMABatch:
    data_tokens: SequenceBatch
    target_tokens: Tensor

    def __del__(self) -> None:
        del self.data_tokens
        del self.target_tokens

class LLaMADataLoader:
    def __init__(
        self,
        text_tokenizer: LLaMATokenizer,
        batching_config: BatchingConfig,
        datapipe: IterDataPipe
    ):
        self.text_tokenizer = text_tokenizer
        self.token_encoder = self.text_tokenizer.create_encoder()
        self.batching_config = batching_config
        self.datapipe = datapipe

    def get_dataloader(self) -> DataLoader:
        data_loader = DataLoader(
            dataset=self.datapipe,
            batch_size=self.batching_config.batch_size,
            shuffle=True,
            collate_fn=self._prepare_batch
        )
        return data_loader

    def __iter__(self) -> Iterable[LLaMABatch]:
        return self.get_dataloader().__iter__()

    def _get_tokenized_target_text(self, sample: str) -> Tensor:
        """Expected sequence is [<eos>, <lang_tok> , ..text tokens.., <eos>]"""
        tokens = self.token_encoder(sample)
        return tokens

    def _batch_tensors(self, tensors: List[Tensor], pad_value: Any) -> Tensor:
        padding_size = max(tensor.shape[0] for tensor in tensors)
        dims = len(tensors[0].shape)
        padded_tensors = []
        for tensor in tensors:
            padding = [0] * 2 * dims
            padding[-1] = padding_size - tensor.shape[0]
            padded_tensors.append(pad_tensor(tensor, padding, "constant", pad_value))
        return torch.stack([tensor for tensor in padded_tensors], dim=0)

    def _prepare_batch(self, samples: List[Dict[str, Any]]) -> LLaMABatch:
        text_tokens_list = [
            self._get_tokenized_target_text(sample) for sample in samples
        ]
        text_pad_idx = self.text_tokenizer.vocab_info.pad_idx
        data_tokens = SequenceBatch(
            self._batch_tensors(
            [tokens[:-1] for tokens in text_tokens_list], pad_value=text_pad_idx
            ).to('cuda'),
            padding_mask=None
        )
        target_tokens = self._batch_tensors(
            [tokens[1:] for tokens in text_tokens_list], pad_value=text_pad_idx
        ).to('cuda')

        return LLaMABatch(
                data_tokens=data_tokens,
                target_tokens=target_tokens
            )