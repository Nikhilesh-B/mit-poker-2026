import torch 
import torch.nn as nn 
import torch.nn.functional as F




class CardEmbedding(nn.Module):
    #dimensionality is for the cards we are talking about whether these are the hole cards or the board cards
    def __init__(self, dim):
        super(CardEmbedding, self).__init__()
        self.rank = nn.Embedding(4, dim)
        self.suit = nn.Embedding(13, dim)
    
    def forward(self, input):
        B, num_cards = input.shape 
        flattened_card = input.view(-1)

        valid = flattened_cards.ge(0).float()
        flattened_cards = flattened_cards.clamp(min=0)

        embs = self.card(flattened_cards)+self.rank()




