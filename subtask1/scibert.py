"""
Implements the `SciBert` model for subtask 1
"""
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding


class SciBert(nn.Module):
    def __init__(self):
        super().__init__()
        # replace pretraining head of the BERT model with a classification head which is randomly initialized
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "allenai/scibert_scivocab_uncased", num_labels=2
        )
        self.collator = collator()

    def forward(self, batch):
        # takes in batch data and passes it through model
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits

    def collate(self, batch):
        return self.collator(batch)

    def predict(self, outputs):
        """
        Maps predicted batch outputs to labels
        """
        predictions = torch.argmax(outputs, dim=1)
        return predictions


class collator:
    """
    Collate function class for DataLoader
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/scibert_scivocab_uncased"
        )
        # predefined collator that pads tensors
        # max length is determined from data exploration
        self.collator = DataCollatorWithPadding(
            self.tokenizer, padding="max_length", max_length=100
        )

    def __call__(self, batch):
        """
        Receives batch data (text, labels) and returns a dictionary containing encoded features, labels, etc.
        """
        texts, labels = zip(*batch)
        texts, labels = list(texts), list(labels)

        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.

        encode = lambda sent: self.tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=100,  # Pad & truncate all sentences.
            padding="max_length",
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
        )

        features = [encode(sent) for sent in texts]
        feature_tensors = self.collator(
            features
        )  # creates a dict containing {'attention_mask', 'input_ids', 'token_type_ids', 'labels'}

        return feature_tensors, torch.LongTensor(labels)
