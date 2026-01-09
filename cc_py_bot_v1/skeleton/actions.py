'''
The actions that the player is allowed to take.
'''
from collections import namedtuple

FoldAction = namedtuple('FoldAction', [])
CallAction = namedtuple('CallAction', [])
CheckAction = namedtuple('CheckAction', [])
# Card should be the index of the card in your hand [0,1,2]
DiscardAction = namedtuple('DiscardAction', ['card'])
# We coalesce BetAction and RaiseAction for convenience
RaiseAction = namedtuple('RaiseAction', ['amount'])
