"""
Implements the `Model2` model for subtask 2
"""
import torch

from torch import nn
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorWithPadding
from transformers.utils.dummy_pt_objects import T5ForConditionalGeneration
from transformers.utils.dummy_sentencepiece_objects import T5Tokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer

class T5(nn.Module):
    """
    A `PyTorch nn.Module` subclass for subtask 2.
    """

    def __init__(self):
        super().__init__()
        self.model = AutoModelWithLMHead.from_pretrained("t5-base")
        self.Collator = Collator()
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")

    def forward(self, x):

        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        target_ids = x['target_ids']
        target_mask = x['target_mask']
        output = self.model(input_ids, attention_mask, target_ids, target_mask)
        return output

    def collate(self, batch):
        """
        Collate function for DataLoader
        """
        return self.Collator(batch)

    def predict(self, batch):
        """
        Given a batch output, predict the final result
        """
        input_ids = batch['input_ids']
        input_mask = batch['attention_mask']
        target_mask = batch['target_mask']

        generated_ids = self.model.generate(input_ids, attention_mask=input_mask, decoder_attention_mask=target_mask)
        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)#outputs.logits #self.tokenizer.decode(outputs.logits, skip_special_tokens=True) 
        return text

    def evaluate(self, preds, labels):
        """
        Evaluates the predicted results against the expected labels and
        returns the tp, fp, tn, fn values for the result batch
        """
        total_match = 0
        pred_length = 0
        label_length = 0
        recall = precision = 0
        for pred, label in zip(preds, labels):
            print(pred)
            for t in pred.tolist():
                #print(label)
                pred_length += 1
                l_encode = self.tokenizer.encode(label)
                print(l_encode)
                for l in l_encode:
                    label_length += 1
                    print("HERE")
                    #print(t)
                    print(l)
                    if int(t) == int(l):
                        total_match += 1

        recall = total_match/label_length
        precision = total_match/pred_length     

        return precision, recall
        
class Collator:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.collator = DataCollatorWithPadding(self.tokenizer)

    def __call__(self, batch):
        articles_seg, c_sents = zip(*batch)
        articles_seg, c_sents = list(articles_seg), list(c_sents)

        # encode = lambda sents: self.tokenizer.batch_encode_plus(       
        #             sent,  # Sentence to encode.
        #             #add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        #             #max_length=100,  # Pad & truncate all sentences.
        #             #padding="max_length",
        #             max_length = 512,
        #             padding="max_length",
        #             truncation=True,
        #             return_attention_mask=True,  # Construct attn. masks.
        #             return_tensors='pt'
        #         )
            
        encode = lambda sents: self.tokenizer.batch_encode_plus(
            sents,
            max_length = 512,
            padding = "max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # in_seq = []
        # out_seq = []
        # for sent in c_sents:
        #     out_seq.append(encode_labels(sent))

        # for idx, article in enumerate(articles_seg):
        #     preprocess_text = ' '.join(article)
        #     encoded = encode(preprocess_text)
        #     encoded['decoder_input_ids'] = out_seq[idx]['input_ids']
        #     in_seq.append(encoded)
        processed_a = []
        for article in articles_seg:
            processed_a.append(' '.join(article))
        features = encode(processed_a)
        targets = encode(c_sents)

        features['input_ids'] = features['input_ids'].squeeze()
        features['attention_mask'] = features['attention_mask'].squeeze()
        features['target_ids'] = targets['input_ids'].squeeze()
        features['target_mask'] = targets['attention_mask'].squeeze()
        print(features)
        return features, targets #self.collator(in_seq), self.collator(out_seq)
