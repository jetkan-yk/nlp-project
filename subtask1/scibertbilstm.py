import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
import pandas as pd 
import transformers
from torch.utils.data import Dataset, DataLoader 

from transformers import AutoTokenizer, AutoModel, BatchEncoding
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.data.data_collator import default_data_collator

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# ref: https://colab.research.google.com/github/sshailabh/SemEval-2021-Task-11/blob/main/Sub-task-A.ipynb#scrollTo=HnNnn2_2-HRn

class SCIBERTBILSTMClass(torch.nn.Module):
    def __init__(self):
        super(SCIBERTBILSTMClass, self).__init__()
        self.l1 = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased',  output_hidden_states=True)
        self.lstm = torch.nn.LSTM(768, 400, num_layers=2, batch_first = True, bidirectional=True)
        self.l2 = torch.nn.Dropout(0.1)
        self.l3 = torch.nn.Linear(800, 400)
        self.l4 = torch.nn.Linear(400,100) # check with layer with 30
        self.l5 = torch.nn.Linear(100,2)
        
        self.collator = collator()
    
    def forward(self, batch):
        ids = batch['input_ids']
        mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        lengths = batch['length'].flatten().cpu()
                
        encoded_layers = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)[2]
        scibert_hidden_layer = encoded_layers[12]
        enc_hiddens, (last_hidden, last_cell) = self.lstm(torch.nn.utils.rnn.pack_padded_sequence(scibert_hidden_layer, lengths, batch_first=True, enforce_sorted=False)) #enforce_sorted=False  #pack_padded_sequence(data and batch_sizes
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)  # (batch_size, 2*hidden_size)
        output_hidden = self.l2(output_hidden)            # no size change
        output_2 = self.l3(output_hidden)
        output_2 = torch.nn.ReLU()(output_2)
        output_4 = self.l4(output_2)
        output_5 = self.l5(output_4)
        
        return output_5
    
    def collate(self, batch):
        return self.collator(batch)

    def predict(self, outputs):
        """
        Maps predicted batch outputs to labels
        """
        return torch.argmax(outputs, dim=1)
    
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/scibert_scivocab_uncased"
        )
        # predefined collator that pads tensors
        # max length is determined from data exploration
        self.collator = DataCollatorWithPadding(
            self.tokenizer, padding="max_length", max_length=100 # max length is decided from data exploration
        )

    def __call__(self, batch):
        """
        Receives batch data (text, labels) and returns a dictionary containing encoded features, labels, etc.
        """
        texts, labels = zip(*batch)
        texts, labels = list(texts), list(labels)
    
        encode = lambda sent: self.tokenizer.encode_plus(sent,
            None,add_special_tokens=True,
            max_length=100,
            padding=True,
            return_token_type_ids=True,
            truncation=True,
            return_length = True
        )
        
        features = [encode(sent) for sent in texts]
                
        feature_tensors = self.collator(
            features
        )  # creates a dict containing {'attention_mask', 'input_ids', 'token_type_ids', 'labels'}

        return feature_tensors, torch.LongTensor(labels)
