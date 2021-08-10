import os
from typing import Optional, List

from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, TensorDataset, DataLoader
from pytorch_lightning import LightningDataModule

from util import get_tokenizer
from util.utils import load_txt


class IntentClassificationDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        intent_vocab: dict,
        mode: str = "train"
    ):
        """
        `IntentClassificationDataset` read data files and convert to tensor dataset.

        Args:
            data_dir (str): path of dataset directory.
            tokenizer (PreTrainedTokenizer): huggingface's PreTrainedTokenizer object.
            max_seq_length (int): the maximum length (in number of tokens) for the inputs to the transformer model.
            intent_vocab (dict): vocabulary of intent.
            mode (str, optional): type of current dataset, e.g. "train", "valid", "test". defaults to "train".
        """
        # load text data and label
        text_data = load_txt(os.path.join(data_dir, mode, "seq.in"))
        intent_data = load_txt(os.path.join(data_dir, mode, "label"))

        # check number of data
        assert len(text_data) == len(intent_data), "text and intent must have the same number of elements!"

        # convert data to features
        features = []
        for idx, (text, intent) in tqdm(enumerate(zip(text_data, intent_data)), total=len(text_data)):
            if tokenizer.__class__.__name__ in [
                "RobertaTokenizer",
                "LongformerTokenizer",
                "BartTokenizer",
                "RobertaTokenizerFast",
                "LongformerTokenizerFast",
                "BartTokenizerFast",
            ]:
                tokens = tokenizer(text, padding='max_length', max_length=max_seq_length, truncation=True, add_prefix_space=True)
            else:
                tokens = tokenizer(text, padding='max_length', max_length=max_seq_length, truncation=True)
            intent_id = intent_vocab[intent]

            if "token_type_ids" in tokens.keys():
                feature = {
                    "input_ids" : tokens["input_ids"],
                    "token_type_ids" : tokens["token_type_ids"],
                    "attention_mask" : tokens["attention_mask"],
                    "intent_id" : intent_id
                }
            else:
                feature = {
                    "input_ids": tokens["input_ids"],
                    "attention_mask": tokens["attention_mask"],
                    "intent_id": intent_id
                }
                feature.update({"token_type_ids": [0]})
            features.append(feature)

        self.features = features

        # convert features to TensorDataset
        all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)
        all_intent_ids = torch.tensor([f["intent_id"] for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_intent_ids)
        self.dataset = dataset

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

class IntentClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        data_name: str,
        model_type: str,
        model_name: str,
        max_seq_length: int = 128,
        batch_size: int = 8
    ):
        """
        `IntentClassificationDataModule` prepare data and data loaders for intent classification.

        Args:
            data_name (str): dataset name, e.g., "atis"
            model_type (str): model type, e.g., "bert"
            model_name (str): name of the specific model, e.g., "bert-base-cased"
            max_seq_length (int, optional): the maximum length (in number of tokens) for the inputs to the transformer model.
                                            defaults to 128.
            batch_size (int, optional): batch size for single GPU. defaults to 8.
        """

        super().__init__()

        # configures
        self.data_name = data_name.lower()
        if self.data_name not in self.get_supported_dataset():
            raise NotImplementedError(self.data_name)

        self.model_type = model_type
        self.model_name = model_name
        self.do_lower_case = True if "uncased" in model_name else False

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        # for balancing between CPU and GPU
        self.num_workers = 4 * torch.cuda.device_count()

    def prepare_data(self) -> None:
        # prepare data directory
        self.data_dir = os.path.join("./", "data", self.data_name)

        # load tokenizer
        self.tokenizer = get_tokenizer(self.model_type, self.model_name, self.do_lower_case)

        # load vocab of intent
        self.intent_vocab = self._load_vocab(os.path.join(self.data_dir, "intent.vocab"))
        self.num_intents = len(self.intent_vocab)

    def _load_vocab(self, file_path: str) -> dict:
        vocab = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                label, id = line.split("\t")
                vocab[label] = int(id)
        return vocab

    def setup(self, stage: Optional[str] = None) -> None:
        # load train/val dataset objects
        if stage == "fit" or stage is None:
            train_dataset = IntentClassificationDataset(self.data_dir, self.tokenizer,
                                                        self.max_seq_length, self.intent_vocab, "train")
            val_dataset = IntentClassificationDataset(self.data_dir, self.tokenizer,
                                                      self.max_seq_length, self.intent_vocab, "valid")
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        # load test dataset objects
        if stage == 'test' or stage is None:
            test_dataset = IntentClassificationDataset(self.data_dir, self.tokenizer,
                                                       self.max_seq_length, self.intent_vocab, "test")
            self.test_dataset = test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @staticmethod
    def get_supported_dataset() -> List[str]:
        """
        This function returns list of supported dataset name by this data module.

        Returns:
            List[str]: list of supported dataset name.
        """
        return ["atis", "snips"]


class EntityRecognitionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        entity_vocab: dict,
        mode: str = "train"
    ):
        """
        `EntityRecognitionDataset` read data files and convert to tensor dataset.

        Args:
            data_dir (str): path of dataset directory.
            tokenizer (PreTrainedTokenizer): huggingface's PreTrainedTokenizer object.
            max_seq_length (int): the maximum length (in number of tokens) for the inputs to the transformer model.
            entity_vocab (dict): vocabulary of entity.
            mode (str, optional): type of current dataset, e.g. "train", "valid", "test". defaults to "train".
        """
        # load text data and label
        text_data = load_txt(os.path.join(data_dir, mode, "seq.in"))
        entities_data = load_txt(os.path.join(data_dir, mode, "seq.out"))

        # check number of data
        assert len(text_data) == len(entities_data), "text and entity must have the same number of elements!"

        # get ignore_index
        pad_token_label_id = CrossEntropyLoss().ignore_index

        # convert data to features
        features = []
        for idx, (text, entities_str) in tqdm(enumerate(zip(text_data, entities_data)), total=len(text_data)):
            tokens = []
            label_ids = []

            words = text.split()
            labels = entities_str.split()
            for word, label in zip(words, labels):
                word_tokens = tokenizer.tokenize(word)
                if not word_tokens:
                    word_tokens = [tokenizer.unk_token]  # For handling the bad-encoded word
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([entity_vocab[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

            special_tokens_count = 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[:(max_seq_length - special_tokens_count)]
                label_ids = label_ids[:(max_seq_length - special_tokens_count)]

            # Add [SEP]
            tokens += [tokenizer.sep_token]
            label_ids += [pad_token_label_id]

            # Add [CLS]
            tokens = [tokenizer.cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids

            token_type_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            attention_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            token_type_ids += [0] * padding_length
            label_ids += [pad_token_label_id] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(attention_mask) == max_seq_length
            assert len(token_type_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            entities = label_ids

            feature = {
                "input_ids" : input_ids,
                "token_type_ids" : token_type_ids,
                "attention_mask" : attention_mask,
                "entities" : entities
            }
            features.append(feature)

        self.features = features

        # convert features to TensorDataset
        all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)
        all_entities = torch.tensor([f["entities"] for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_entities)
        self.dataset = dataset

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

class EntityRecognitionDataModule(LightningDataModule):
    def __init__(
        self,
        data_name: str,
        model_type: str,
        model_name: str,
        max_seq_length: int = 128,
        batch_size: int = 8
    ):
        """
        `EntityRecognitionDataModule` prepare data and data loaders for entity recognition.

        Args:
            data_name (str): dataset name, e.g., "atis"
            model_type (str): model type, e.g., "bert"
            model_name (str): name of the specific model, e.g., "bert-base-cased"
            max_seq_length (int, optional): the maximum length (in number of tokens) for the inputs to the transformer model.
                                            defaults to 128.
            batch_size (int, optional): batch size for single GPU. defaults to 8.
        """

        super().__init__()

        # configures
        self.data_name = data_name.lower()
        if self.data_name not in self.get_supported_dataset():
            raise NotImplementedError(self.data_name)

        self.model_type = model_type
        self.model_name = model_name
        self.do_lower_case = True if "uncased" in model_name else False

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        # for balancing between CPU and GPU
        self.num_workers = 4 * torch.cuda.device_count()

    def prepare_data(self) -> None:
        # prepare data directory
        self.data_dir = os.path.join("./", "data", self.data_name)

        # load tokenizer
        self.tokenizer = get_tokenizer(self.model_type, self.model_name, self.do_lower_case)

        # load vocab of entity
        self.entity_vocab = self._load_vocab(os.path.join(self.data_dir, "entity.vocab"))
        self.num_entities = len(self.entity_vocab)

    def _load_vocab(self, file_path: str) -> dict:
        vocab = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                label, id = line.split("\t")
                vocab[label] = int(id)
        return vocab

    def setup(self, stage: Optional[str] = None) -> None:
        # load train/val dataset objects
        if stage == "fit" or stage is None:
            train_dataset = EntityRecognitionDataset(self.data_dir, self.tokenizer,
                                                     self.max_seq_length, self.entity_vocab, "train")
            val_dataset = EntityRecognitionDataset(self.data_dir, self.tokenizer,
                                                   self.max_seq_length, self.entity_vocab, "valid")
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        # load test dataset objects
        if stage == 'test' or stage is None:
            test_dataset = EntityRecognitionDataset(self.data_dir, self.tokenizer,
                                                    self.max_seq_length, self.entity_vocab, "test")
            self.test_dataset = test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @staticmethod
    def get_supported_dataset() -> List[str]:
        """
        This function returns list of supported dataset name by this data module.

        Returns:
            List[str]: list of supported dataset name.
        """
        return ["atis", "snips"]