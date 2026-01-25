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
        flattened_cards = input.view(-1)

        valid_flattened_cards = flattened_cards.ge(0).float() #-1 means no card we can force the hand to be 3 and the board to be 6 and then with -1 in the missing spots
        flattened_cards = flattened_cards.clamp(min=0)

        embs = self.card(flattened_cards)+self.rank(flattened_cards//10)+self.suit(flattened_cards%10)
        embs = embs*valid_flattened_cards.unsqueeze(1)

        return embs.view(B, num_cards, -1).sum(1)

    
class 




