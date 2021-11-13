"""
Implements the `Model2` model for subtask 2
"""
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

START_TAG = "<START>"
STOP_TAG = "<STOP>"

class SciBert_BiLSTM_CRF(nn.Module):
    def __init__(self, tag_to_ix, embedding_dim, hidden_dim):
        super(SciBert_BiLSTM_CRF, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
 
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        #Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        #Pretrained SciBert downloaded from allenai
        self.modell = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

        #Matrix of transition parameters. Entry i,j is the score of transitioning to i from j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        #These two statements enforce the constraint that we never transfer to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self): 
        return (torch.randn(2, 1, self.hidden_dim // 2).to(device),
                torch.randn(2, 1, self.hidden_dim // 2).to(device))

    def _forward_alg(self, feats):
        #Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device)
        #START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        #Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        #Iterate through the sentence
        for feat in feats:
            alphas_t = []  #The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                #Broadcast the emission score: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                #The ith entry of trans_score is the score of transitioning to next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                #The ith entry of next_tag_var is the value for th edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the scores.
                alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = self.log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        #The tokenized sentence is passed through SciBERT and output from LSTM is mapped into tag space
        outputs = self.modell(**sentence, output_hidden_states = True)
        scibert_out = ((outputs[2][12])[0]).view(len(sentence["input_ids"][0]), 1, -1)
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(scibert_out, self.hidden)
        lstm_out = lstm_out.view(len(sentence["input_ids"][0]), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        #Gives the score of a provided tag sequence
        score = torch.zeros(1).to(device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        #Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        #forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  #Holds the backpointers for this step
            viterbivars_t = [] #Holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                #next_tag_var[i] holds the viterbi variable for tag i at the previous step, plus the score of transitioning
                #from tag i to next_tag. We don't include the emission scores here because the max does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            #Now we add in the emission scores, and assign forward_var to the set of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        #Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = self.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        #Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        #Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  #Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        #The loss function 
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  
        #Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        #Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    #### Helper Functions

    #Return the argmax as a python int
    def argmax(vec):
        _, idx = torch.max(vec, 1)
        return idx.item()

    #The function uses SciBERT tokenizer to prepare the sentence to be fed into SciBERT 
    def prepare_sequence(seq):
        for count,i in enumerate(seq):
        temp = self.tokenizer.tokenize(i) 
        if(len(temp)>1): #Only the first token of the word has been considered(same as in NER paper) 
            seq[count] = temp[0]           
        sentences = " ".join(seq)
        inputs = self.tokenizer(sentences, return_tensors="pt")
        return inputs


    # Compute log sum exp in a numerically stable way for the forward algorithm
    def log_sum_exp(vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
            torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))