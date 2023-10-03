import os
import torch
import random
from typing import List
import torch.utils.data as data
from torch.utils.data import DataLoader
from tokenizers import ByteLevelBPETokenizer

# ===================================================================

SPEC_TOKENS = {
    "[PAD]": 0,
    "[SEP]": 1,
    "[CLS]": 2,
    "[UNK]": 3,
}

def get_datasplits(path:str, batch_size:int=16, train_split:float=0.8) -> (DataLoader, DataLoader):
    raw_sents = RawCorrSents.read_from_file(path)
    train_sents, test_sents = RawCorrSents.get_splits(raw_sents, train_split=train_split)

    train_ds = CorrSentsDataset(tokenizer_source=path, samples=train_sents)
    val_ds = CorrSentsDataset(tokenizer_source=path, samples=test_sents, tokenizer=train_ds.tokenizer)
    assert len(train_ds) > 0
    assert len(val_ds) > 0

    train_dl = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)

    return train_dl, val_dl, train_ds.tokenizer

def merge(sentences:List[object], pad_token) -> (torch.LongTensor, torch.BoolTensor):
    lengths = [sent.shape[0] for sent in sentences]
    maxlen = max(1, max(lengths))
    padded_ids = torch.LongTensor(len(sentences), maxlen).fill_(pad_token)
    padding_masks = torch.BoolTensor(len(sentences), maxlen).fill_(True)
    for i, sample in enumerate(sentences): 
        padded_ids[i, :sample.shape[0]] = sample # right padding
        padding_masks[i, :sample.shape[0]] = False
    return padded_ids, padding_masks

def collate_fn(batch:List[object]) -> object:
    sents = [sample['sentence'] for sample in batch]
    corruptions = [sample['corruption'] for sample in batch]
    X = []
    X_seg = []
    Y = []
    for i in range(len(batch)):
        # <s> sent </s><s> corruption </s>
        # random shuffling to avoid shortcut learning by position
        if random.randint(0, 1):
            X.append(torch.LongTensor([SPEC_TOKENS["[CLS]"]] + sents[i].ids + [SPEC_TOKENS["[SEP]"]] + corruptions[i].ids))
            X_seg.append(torch.LongTensor([0] + [0] * len(sents[i].ids) + [0] + [1] * len(corruptions[i].ids)))
            Y.append(0)
        else:
            X.append(torch.LongTensor([SPEC_TOKENS["[CLS]"]] + corruptions[i].ids + [SPEC_TOKENS["[SEP]"]] + sents[i].ids))
            X_seg.append(torch.LongTensor([0] + [1] * len(corruptions[i].ids) + [0] + [0] * len(sents[i].ids)))
            Y.append(1)

    X, X_mask = merge(X, pad_token=SPEC_TOKENS["[PAD]"])
    X_seg, _ = merge(X_seg, pad_token=SPEC_TOKENS["[PAD]"])
    Y = torch.LongTensor(Y)
    return {"X": X, "X_mask": X_mask, "X_seg": X_seg, "Y": Y}

# ===================================================================

# https://huggingface.co/blog/how-to-train
class CorruptionsTokenizer():
    def __init__(self, path:str, save_path:str, vocab_size:int=52000):
        if os.path.exists(save_path):
            self.tokenizer = ByteLevelBPETokenizer(
                vocab=os.path.join(save_path, "vocab.json"),
                merges=os.path.join(save_path, "merges.txt")
            )
        else:
            self.tokenizer.train(files=[path], vocab_size=vocab_size, min_frequency=1, special_tokens=[
                "[PAD]",
                "[SEP]",
                "[CLS]",
                "[UNK]",
            ])
            self.tokenizer.save_model(save_path)
        self.tokenizer.enable_truncation(max_length=512)

    def __call__(self, text:str) -> object:
        return self.tokenizer.encode(text)
    
    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()

# ===================================================================

class CorrSentsDataset(data.Dataset):
    
    def __init__(self, tokenizer_source, samples, tokenizer=None):
        self.samples = samples

        if tokenizer is None:
            self.tokenizer = CorruptionsTokenizer(
                path=tokenizer_source, 
                save_path=os.path.join("models", "tokenizer")
            )
        else: self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx:int) -> object:
        return {'sentence': self.tokenizer(self.samples[idx][0]), 'corruption': self.tokenizer(self.samples[idx][1])}


class RawCorrSents():

    @staticmethod
    def get_splits(samples:List[str], train_split:float=0.8):
        train_idx = int(len(samples)*train_split)
        return samples[:train_idx], samples[train_idx:]
    
    @staticmethod
    def read_from_file(path:str) -> List[str]:
        with open(path, "r", encoding="utf8") as f:
            samples = [line.replace("\n", "").split("\t") for line in f.readlines()]
        return samples