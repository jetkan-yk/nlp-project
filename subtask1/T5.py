"""
Implements the `Model2` model for subtask 2
"""
import torch

from torch import nn
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorWithPadding
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

    def forward(self, x):

        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        decoder_inputs = x['decoder_input_ids'].contiguous()
        output = self.model(input_ids, attention_mask, decoder_inputs)
        #output = self.model.generate(input_ids, max_length=20)
        return output

    def collate(self, batch):
        """
        Collate function for DataLoader
        """
        return self.Collator(batch)

    def predict(self, outputs):
        """
        Given a batch output, predict the final result
        """
        return outputs

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
            for t in pred.tolist():
                pred_length += 1
                for l in label.tolist():
                    label_length += 1
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

        encode = lambda sent: self.tokenizer.encode_plus(       
                    sent,  # Sentence to encode.
                    #add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    #max_length=100,  # Pad & truncate all sentences.
                    #padding="max_length",
                    max_length = 512,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,  # Construct attn. masks.
                )
            
        encode_labels = lambda sent: self.tokenizer.encode_plus(
            sent,
            max_length = 512,
            padding = "max_length",
            truncation=True,
            return_attention_mask=False
        )

        in_seq = []
        out_seq = []
        for sent in c_sents:
            out_seq.append(encode_labels(sent))

        for idx, article in enumerate(articles_seg):
            preprocess_text = ' '.join(article)
            encoded = encode(preprocess_text)
            encoded['decoder_input_ids'] = out_seq[idx]['input_ids']
            in_seq.append(encoded)

        return self.collator(in_seq), self.collator(out_seq)
