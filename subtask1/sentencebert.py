import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
import pandas as pd 
import transformers
from torch.utils.data import Dataset, DataLoader 

# code taken from https://victordibia.com/blog/extractive-summarization/

from transformers import AutoTokenizer, AutoModel

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


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
    def __init__(self, model_name="sentence-transformers/paraphrase-MiniLM-L3-v2", in_features=384):
        super(SentenceBertClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(in_features*3, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)
        self.classifierSigmoid = torch.nn.Sigmoid()
        
        self.collator = collator()

#     def forward(self, sent_ids, doc_ids, sent_mask, doc_mask):
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

        # Concatenate input features and their elementwise product
        concat_features = torch.cat((sentence_embeddings, doc_embeddings, combined_features), dim=1)   
        
        pooler = self.pre_classifier(concat_features) 
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.classifierSigmoid(output) 

        return output
    
    def collate(self, batch):
        return self.collator(batch)

    def predict(self, outputs):
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
        sentence_model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(sentence_model_name) 
        self.max_len = 512
        
    def __call__(self, batch):
        """
        Receives batch data (docs, doc_labels) and returns a dictionary containing encoded features, labels, etc.
        """
        docs, doc_labels = zip(*batch)
        # docs is [sents in doc], doc_labels is [labels for each sent in doc]
        docs, doc_labels = list(docs), list(doc_labels)
        
        # creates sentence, document pairs
        texts = []
        for doc in docs:
            for sent in doc:
                texts.append([" ".join(sent.split()), " ".join(doc)])
        
        def encode(sentence, document):
            # encodes [sentence, document]
            
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.

            inputs = self.tokenizer.batch_encode_plus(
                [sentence, document], 
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True
            )
            ids = inputs['input_ids']
            mask = inputs['attention_mask']

            return {
                'sent_ids': torch.tensor(ids[0], dtype=torch.long),
                'doc_ids': torch.tensor(ids[1], dtype=torch.long),
                'sent_mask': torch.tensor(mask[0], dtype=torch.long),
                'doc_mask': torch.tensor(mask[1], dtype=torch.long),
            } 
        
        encoded_texts = [encode(sentence, document) for sentence, document in texts]
        
        # converts list of encoded input dicts into dictionary of encoded inputs
        sent_ids = torch.tensor([encoded_text['sent_ids'] for encoded_text in encoded_texts]).to(device, dtype = torch.long)
        doc_ids = torch.tensor([encoded_text['doc_ids'] for encoded_text in encoded_texts]).to(device, dtype = torch.long)
        sent_mask = torch.tensor([encoded_text['sent_mask'] for encoded_text in encoded_texts]).to(device, dtype = torch.long)
        doc_mask = torch.tensor([encoded_text['doc_mask'] for encoded_text in encoded_texts]).to(device, dtype = torch.long)
        targets = torch.tensor(doc_labels, dtype=torch.long).to(device, dtype = torch.float)
        
        batch = {
                'sent_ids': sent_ids,
                'doc_ids': doc_ids,
                'sent_mask': sent_mask,
                'doc_mask': doc_mask,
            } 
        
        return batch, targets

        
#     def __getitem__(self, index):
#         # gets sentence in string
#         sentence = str(self.data.iloc[index].sents)
#         sentence = " ".join(sentence.split())
        
#         # gets document in string
#         document = str(self.data.iloc[index].docs)
#         document = " ".join(document.split())

#         # encodes [sentence, document]
#         inputs = self.tokenizer.batch_encode_plus(
#             [sentence, document], 
#             add_special_tokens=True,
#             max_length=self.max_len,
#             padding="max_length",
#             return_token_type_ids=True,
#             truncation=True
#         )
#         ids = inputs['input_ids']
#         mask = inputs['attention_mask']

#         return {
#             'sent_ids': torch.tensor(ids[0], dtype=torch.long),
#             'doc_ids': torch.tensor(ids[1], dtype=torch.long),
#             'sent_mask': torch.tensor(mask[0], dtype=torch.long),
#             'doc_mask': torch.tensor(mask[1], dtype=torch.long),
#             'targets': torch.tensor([self.data.iloc[index].y], dtype=torch.long)
#         } 
    
#     def __len__(self):
#         return self.len