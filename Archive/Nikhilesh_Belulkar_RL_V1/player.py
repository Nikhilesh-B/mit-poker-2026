'''
Simple example pokerbot, written in Python.
'''
import os
import random
import torch

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

from training.inference import load_policy, encode_obs, build_mask, map_action_to_engine


class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_path = os.environ.get("POKERBOT_CHECKPOINT", "checkpoints/nfsp_ep1000.pt")
        self.policy = None
        try:
            self.policy, self.policy_cfg = load_policy(ckpt_path, device=self.device)
            print(f"Loaded policy from {ckpt_path}")
        except FileNotFoundError:
            print(f"Checkpoint not found at {ckpt_path}, using fallback random policy")
        self.last_action = None

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        # the total number of seconds your bot has left to play this game
        game_clock = game_state.game_clock
        round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        my_cards = round_state.hands[active]  # your cards
        big_blind = bool(active)  # True if you are the big blind
        self.last_action = None

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        street = previous_state.street  # 0,2,3,4,5,6 representing when this round ended
        my_cards = previous_state.hands[active]  # your cards
        opp_cards = previous_state.hands[1-active]         # opponent's cards or [] if not revealed
        self.last_action = None

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        # If no trained policy available, fall back to basic random logic
        if self.policy is None:
            legal_actions = round_state.legal_actions()
            if DiscardAction in legal_actions:
                return DiscardAction(0)
            if RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                if random.random() < 0.5:
                    return RaiseAction(min_raise)
            if CheckAction in legal_actions:
                return CheckAction()
            if random.random() < 0.25:
                return FoldAction()
            return CallAction()

        obs_t = encode_obs(round_state, active, self.last_action).to(self.device)
        legal_mask = build_mask(round_state, active).to(self.device)
        if legal_mask.sum() <= 0:
            return CheckAction()
        with torch.no_grad():
            probs = self.policy(obs_t.unsqueeze(0), legal_mask.unsqueeze(0)).squeeze(0)
            action_id = int(probs.argmax().item())
        self.last_action = action_id
        return map_action_to_engine(action_id, round_state)


if __name__ == '__main__':
    run_bot(Player(), parse_args())
