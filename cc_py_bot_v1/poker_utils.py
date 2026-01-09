from collections import Counter

from math import sqrt, log

RANK_MAP = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, 
             '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, 
             '4': 4, '3': 3, '2': 2}
RANKS = set(RANK_MAP.values())

SUIT_MAP = {'h': 0, 's': 1, 'd': 2, 'c': 3}
SUITS = set(SUIT_MAP.values())

HAND_RANKS = {'HighCard': 0, 'Pair': 1, 'TwoPair': 2, 
              'ThreeOfAKind': 3, 'Straight': 4, 'Flush': 5, 
              'FullHouse': 6, 'FourOfAKind': 7, 'StraightFlush': 8}

# Straight Helpers
def has_straight(rank_set):
    if 14 in rank_set:
        rank_set = rank_set | {1}
    for start in range(1, 11):
        if all(start + i in rank_set for i in range(5)):
            return True
    return False

def straight_draw(rank_set):
    if 14 in rank_set:
        rank_set = rank_set | {1}
    for start in range(1, 11):
        hits = sum((start + i) in rank_set for i in range(5))
        if hits == 4:
            return True
    return False


def evaluate_poker_hands(hand_cards, board_cards, MAX_CARDS=8):
    cards = hand_cards + board_cards
    
    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]
    
    rank_counter = Counter(ranks)
    suit_counter = Counter(suits)

    unique_ranks = set(ranks)
    
    result = {'Pair': 0, 'TwoPair': 0, 'ThreeOfAKind': 0,
              'Straight': 0, 'Flush': 0, 'FullHouse': 0, 
              'FourOfAKind': 0, 'StraightFlush': 0}
    
    counts = sorted(rank_counter.values(), reverse=True)
    
    # ** Made Hands
    # Counting Hands
    if counts[0] >= 2:
        result['Pair'] = 2
    if (len(counts) >= 2) and ((counts[0] >= 2) and (counts[1] >= 2)):
        result['TwoPair'] = 2
    if counts[0] >= 3:
        result['ThreeOfAKind'] = 2
    if counts[0] >= 4:
        result['FourOfAKind'] = 2
    if (len(counts) >= 2) and ((counts[0] >= 3) and (counts[1] >= 2)):
        result['FullHouse'] = 2
    
    # Straight
    if has_straight(unique_ranks):
        result['Straight'] = 2
    
    # Flush
    flush_suits = [s for s, c in suit_counter.items() if c >= 5]
    if flush_suits:
        result['Flush'] = 2
        
    # Straight Flush
    if result['Straight'] == 2 and result['Flush'] == 2:
        for suit in flush_suits:
            suited_ranks = {r for r, s in cards if s == suit}
            if has_straight(suited_ranks):
                result['StraightFlush'] = 2
                break
    
    # ** Draws
    if len(cards) < MAX_CARDS:
        # Counting Draws
        if result['Pair'] == 0:
            result['Pair'] = 1
        if (result['TwoPair'] == 0) and (counts[0] == 2
                                         ) and (len(cards) >= 3):
            result['TwoPair'] = 1
        if (result['ThreeOfAKind'] == 0) and (counts[0] == 2):
            result['ThreeOfAKind'] = 1
        if (result['FourOfAKind'] == 0) and (counts[0] == 3):
            result['FourOfAKind'] = 1
        
        # Flush Draw
        if result['Flush'] == 0:
            flush_suits_d = [s for s, c in suit_counter.items() if c >= 4]
            if flush_suits_d:
                result['Flush'] = 1
        
        # Straight Draw
        if result['Straight'] == 0:
            if straight_draw(unique_ranks):
                result['Straight'] = 1
        
        # Full House Draw
        if result['FullHouse'] == 0:
            # Trips + another rank
            trips_ranks = [r for r, c in rank_counter.items() if c >= 3]
            if trips_ranks and len(rank_counter) >= 2:
                result['FullHouse'] = 1
            else:
                # Two or more pairs (or pairs + trips not already made)
                pair_ranks = [r for r, c in rank_counter.items() if c >= 2]
                if len(pair_ranks) >= 2:
                    result['FullHouse'] = 1
                
        # Straight flush draw
        if result['StraightFlush'] == 0:
            for suit, cnt in suit_counter.items():
                if cnt >= 4:
                    suited_ranks = {r for r, s in cards if s == suit}
                    if straight_draw(suited_ranks):
                        result['StraightFlush'] = 1
                        break
    return result

def poker_hand_scorer(hand_cards, board_cards):
    ''' Heuristic evaluator. '''
    
    score = 0
    poker_hand = evaluate_poker_hands(hand_cards, board_cards)
    
    cards = hand_cards + board_cards
    
    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]
    
    rank_counter = Counter(ranks)
    suit_counter = Counter(suits)
    
    # High Card
    for r in ranks:
        score += 0.25*r
    
    # Counts- for scoring Pair, TwoPair, 3oaK, 4oaK, FH
    for r, c in rank_counter.items():
        score += sqrt(r)*c**2
    
    # Hand Ranking Premiums
    for k, v in poker_hand.items():
        score += HAND_RANKS[k]*v
        
    # Flush and Straight
    if poker_hand['Straight'] == 2:
        score += 10*sqrt(max(ranks))
    elif poker_hand['Straight'] == 1:
        score += 7*sqrt(max(ranks))
    if poker_hand['Flush'] == 2:
        score += 12*sqrt(max(ranks))
    elif poker_hand['Flush'] == 1:
        score += 8*sqrt(max(ranks))
    if poker_hand['StraightFlush'] == 2:
        score += 20*sqrt(max(ranks))
    elif poker_hand['StraightFlush'] == 1:
        score += 10*sqrt(max(ranks))
        
    # Normalize by number of cards.
    score /= len(cards)
        
    return score
    