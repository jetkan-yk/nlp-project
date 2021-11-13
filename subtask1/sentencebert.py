import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
import pandas as pd 
import transformers
from torch.utils.data import Dataset, DataLoader 

# ref: https://victordibia.com/blog/extractive-summarization/

from transformers import AutoTokenizer, AutoModel, BatchEncoding
from transformers.data.data_collator import default_data_collator

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

SENTENCE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# get mean pooling for sentence bert models 
# ref https://www.sbert.net/examples/applications/computing-embeddings/README.html#sentence-embeddings-with-transformers
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 
# Note that different sentence transformer models may have different in_feature sizes
class SentenceBertClass(torch.nn.Module):
    def __init__(self, model_name="sentence-transformers/paraphrase-MiniLM-L3-v2", in_features=768):
        super(SentenceBertClass, self).__init__()
        model_name = SENTENCE_MODEL_NAME
        self.l1 = AutoModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(in_features*3, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)
        self.classifierSigmoid = torch.nn.Sigmoid()
        
        self.collator = collator()

    def forward(self, batch):
        # takes in batch data and passes it through model
        sent_ids = batch["sent_ids"]
        doc_ids = batch["doc_ids"]
        sent_mask = batch["sent_mask"]
        doc_mask = batch["doc_mask"]
                
        sent_output = self.l1(input_ids=sent_ids, attention_mask=sent_mask) 
        sentence_embeddings = mean_pooling(sent_output, sent_mask) 

        doc_output = self.l1(input_ids=doc_ids, attention_mask=doc_mask) 
        doc_embeddings = mean_pooling(doc_output, doc_mask)

        # elementwise product of sentence embs and doc embs
        combined_features = sentence_embeddings * doc_embeddings  

        # concatenate input features and their elementwise product
        concat_features = torch.cat((sentence_embeddings, doc_embeddings, combined_features), dim=1)   
        
        pooler = self.pre_classifier(concat_features) 
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.classifierSigmoid(output).flatten() 
        
        return output
    
    def collate(self, batch):
        return self.collator(batch)

    def predict(self, outputs):
        # outputs is a list of scores
        # to convert into a label, we take any score > threshold as 1 and 0 otherwise
        threshold = 0.6
        outputs = torch.tensor([1 if score > threshold else 0 for score in outputs])
        return outputs
    
    def evaluate(self, preds, labels):
        """
        Evaluates the predicted results against the expected labels and
        returns the tp, fp, tn, fn values for the result batch
        """
        tp = fp = tn = fn = 0

        for pred, label in zip(preds, labels):
            if pred == 1 and label == 1:
                tp += 1
            if pred == 1 and label == 0:
                fp += 1
            if pred == 0 and label == 1:
                fn += 1
            if pred == 0 and label == 0:
                tn += 1

        return tp, fp, tn, fn
    
    
class collator:
    """
    Collate function class for DataLoader
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(SENTENCE_MODEL_NAME) 
        self.max_len = 384 # max sequence length of model
        
    def __call__(self, batch):
        """
        Receives batch data (docs, doc_labels) and returns a dictionary containing encoded features, labels, etc.
        """
        docs_sents, labels = zip(*batch)
        # docs is [sents in doc], doc_labels is [labels for each sent in doc]
        docs_sents, labels = list(docs_sents), list(labels)
        
        def encode(sentence, document):
            # encodes [sentence, document]
            
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.

            batch_encoding = self.tokenizer.batch_encode_plus(
                [sentence, document], 
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True
            )
            ids = batch_encoding['input_ids']
            mask = batch_encoding['attention_mask']
            
            return {
                'sent_ids': torch.tensor(ids[0], dtype=torch.long),
                'doc_ids': torch.tensor(ids[1], dtype=torch.long),
                'sent_mask': torch.tensor(mask[0], dtype=torch.long),
                'doc_mask': torch.tensor(mask[1], dtype=torch.long),
            } 
        
        encoded_texts = [encode(sentence, document) for sentence, document in docs_sents]
        
        # collates batches of dict-like objects
        # creates a dict containing {'sent_ids', 'doc_ids', 'sent_mask', 'doc_mask'}
        batch = BatchEncoding(default_data_collator(encoded_texts))
        
        targets = torch.tensor(labels, dtype=torch.float)

        return batch, targets