from torch.utils.data import Dataset
import torch

class ProductLabelsDataset(Dataset):
    def __init__(self, dataframe, tokenizer,source_len = 64, target_len = 32):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.source_len = source_len
        self.target_len = target_len

        self.source_text = []
        for _, row in self.dataframe.iterrows():
            self.source_text.append(" | ".join([row["a1"], row["a2"], row["a3"], row["a4"]]))
        self.target_text = self.dataframe["caption"].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length = self.source_len,
            pad_to_max_length = True,
            truncation = True,
            padding = "max_length",
            return_tensors = "pt"
        )

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length = self.target_len,
            pad_to_max_length = True,
            truncation = True,
            padding = "max_length",
            return_tensors = "pt"
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }

