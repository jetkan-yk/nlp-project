import math
from config import Pipeline
from torch.utils.data import Dataset


class Dataset1(Dataset):
    """
    A `PyTorch Dataset` class that accepts a subtask number and a data directory.

    For subtask 1:
        x = a full article plaintext (a list of `string`) \\
        y = a contributing sentence (a `string`)
    """

    def __init__(self, names, articles, sents, phrases, config):
        """
        Initializes the dataset for subtask 1
        """
        self.names = names
        self.articles = articles
        self.sents = sents
        self.phrases = phrases

        self.x = []
        self.y = []

        if config["PIPELINE"] is Pipeline.CLASSIFICATION:
            # formats data for classification task (sent, label),
            # where label == 1 for contributing sentences and label == 0 otherwise
            for idx, sent_list in enumerate(self.sents):
                article = self._stringify(idx)
                for sent_id, sent in enumerate(article):
                    label = int(sent_id in sent_list)
                    self.x.append(sent)
                    self.y.append(label)
                    
        elif config["PIPELINE"] is Pipeline.SBERTEXTRACTIVE:
            # [doc, sent, label]
            for idx, sent_list in enumerate(self.sents):
                article = self._stringify(idx)
                for sent_id, sent in enumerate(article):
                    label = int(sent_id in sent_list)
                    self.x.append([" ".join(article), sent])
                    self.y.append(label)
                    
        elif config["PIPELINE"] is Pipeline.EXTRACTIVE:
            # formats data for extractive summarization task
            # [[sents in document], [labels]] 
            # label == 1 for contributing sentences and label == 0 otherwise
            for idx, sent_list in enumerate(self.sents):
                article = self._stringify(idx)
                batch_x = []
                batch_y = []
                for sent_id, sent in enumerate(article):
                    label = int(sent_id in sent_list)
                    batch_x.append(sent)
                    batch_y.append(label)
                
                self.x.append(batch_x)
                self.y.append(batch_y)
        elif config["PIPELINE"] is Pipeline.SUMMARISATION:

            #process data in this manner:
            #avg no. of tokens / sent = 20
            #to fit in 512 architecutre ~ 25 sentences per segment cos need add title to the sentence
            #segment article into parts --> about 25 sentences per segment 
            #some segments might not have any contributing sentences
            #should return x = ([CLS] + title + sent + [SEP]) of all 15 sentences concatenated into a flat sent
            # y = array of length 15 each positioned demarked with 0/1 to show if the sentence is a contributing sentence
            
            for idx, sent_list in enumerate(self.sents):
                #stringify article
                article = self._stringify(idx)
                #extract title of paper located at index 1
                #title appended at the start of every segment
                title = article[1]
                total_segs = math.ceil(len(article)/25)
                
                for i in range(0, total_segs):
                    seg = []
                    labels = []
                    for j in range(0, 25):
                        #check if its in the last segment
                        sent_idx = i * 25 + j
                        if sent_idx < len(article):
                            # processed_sent = '[CLS] ' + article[sent_idx]  + ' [SEP]'
                            # seg.append(processed_sent)
                            seg.append(article[sent_idx])
                            #if is contributing sentence, label as 1 else as 0
                            #labels.append(1 if sent_idx in sent_list else 0)
                            
                            #if is contributing sentence, add idx to labels
                            #index re positioned to start at 0 respective to the article segment
                            if sent_idx in sent_list:
                                labels.append(self._stringify((idx, sent_idx)))
                    #append title of paper to the start
                    #output is as such: x is a list of sentences with title as the first sent in the segment
                    #y is a list of all the indexes in this segment that are contributing sentences --> zeroed relatively to the index
                    seg.append(title)

                    #if length of labels = 0 then skip over the segment cos it does not have any contributing sentences
                    if len(labels) != 0:
                        self.x.append(seg)
                        #print(labels)
                        self.y.append(' '.join(labels))    
        else:
            raise NotImplementedError

    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return len(self.x)

    def __getitem__(self, i):
        """
        Returns the i-th sample's `(x, y)` tuple
        """
        return self.x[i], self.y[i]

    def _stringify(self, data):
        """
        Converts the article or sentence index into `string`
        """
        if isinstance(data, list):
            return data
        elif isinstance(data, tuple):
            idx, sent = data
            return self.articles[idx][sent]
        elif isinstance(data, int):
            return self.articles[data]
        else:
            raise KeyError
