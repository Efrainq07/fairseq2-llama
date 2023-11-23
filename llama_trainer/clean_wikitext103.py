from typing import Union, Tuple

from torchtext.datasets import WikiText103
from torchdata.datapipes.iter import IterDataPipe

class CleanWikiText103(IterDataPipe):
    def __init__(
            self,
            split: str
            ) -> None:
        super().__init__()
        self.datapipe = WikiText103(split=(split,))[0]

    def __iter__(self):
        for sample in self.datapipe:
            if not sample.startswith(' =') and sample != ' \n':
                yield sample