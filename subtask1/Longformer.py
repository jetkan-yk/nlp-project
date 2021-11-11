"""
Implements the `Model1` class and other subtask 1 helper functions
"""
import torch
from torch import nn
from transformers import DataCollatorWithPadding
from transformers import LongformerModel, LongformerTokenizer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Longformer(nn.Module):
    """
    A `PyTorch nn.Module` subclass for subtask 1.
    """

    def __init__(self):
        super().__init__()
        #init model with default configuration --> 4096 seq length + 512 attention size
        self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.Collator = Collator()

    def forward(self, x):
        batch = x
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        print(input_ids)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def collate(self, batch):
        """
        Collate function for DataLoader
        """
        return self.Collator(batch)

    def predict(self, outputs):
        """
        Given a batch output, predict the final result
        """
        #outputs will be a tokens of the contributing sentences all flattened into 1 giant sent
        #need to split them up into indivi sentences again and untokenise them 
        #but how
        return outputs


class Collator:

    def __init__(self):
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')#AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.collator = DataCollatorWithPadding(self.tokenizer)

    def __call__(self, batch):
        articles, c_sents = zip(*batch)
        articles, c_sents = list(articles), list(c_sents)
       
       #take the articles, prepend and append [cls] [sep] tokens to each sentence
        #flatten every article and every c_sents into a giant string
        #tokenise the giant string

        #define encoding function
        encode = lambda sent: self.tokenizer.encode_plus(
                sent, # Sentence to encode.
                #add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                padding = 'max_length',
                truncation = True,
                return_attention_mask = True,   # Construct attn. masks.
                )


        in_seq = []
        for article in articles:
            temp = []
            for sent in article:
                temp.append(('[CLS] ' + sent + ' [SEP]'))
            #flatten all sentences in the article into 1 string
            flat_article = ' '.join(temp)
            #encode entire article
            print(flat_article)
            in_seq.append(encode(flat_article))

        #print(self.pad_tensor(c_sents))
        
        return self.collator(in_seq), torch.LongTensor(self.pad_arrs(c_sents)) #self.pad_tensor(c_sents)

    def pad_arrs(self, data):
        max_length = 0
        for arr in data:
            size = len(arr)
            if(size > max_length):
                max_length = size

        out = []
        for arr in data:
            size = len(arr)
            tail = [-1] * (max_length - size)
            out.append(arr + tail)
        
        return out



        
        
        


        

