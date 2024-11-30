import torch
import random
import numpy as np
import os
import sys
from collections import deque
from PIL import ImageGrab
import numpy as np
import json
from texasholdem.game.game import TexasHoldEm
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

BUYIN = 500
BIG_BLIND = 5
SMALL_BLIND = 2
MAX_PLAYERS = 2
INPUT_SIZE = 7 + 3*(MAX_PLAYERS-1)

class Game:
    def __init__(self, buyin, big_blind, small_blind, max_players):
        self.game = TexasHoldEm(buyin=buyin, big_blind=big_blind, small_blind=small_blind, max_players=max_players)


    def play_step(self):

        #

        return reward, done, score


class Agent:

    def __init__(self):
        self.n_hands = 0
        self.n_games = 0
        self.epslilon = 0 #randomness
        self.gamma = 0 # discount rate
        self.hand_memory = deque(maxlen=MAX_MEMORY)
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(INPUT_SIZE, 256, 1, 1)

    def get_state(self):
        # cards on board
        # cards in hand
        # action of other players

    def short_remember(self, state, action, reward, next_state, done):
        self.hand_memory.append((state, action, reward, next_state, done))
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def get_action(self, state, big_blind, min_bet):

        """
        prediction
        if (prediction <= -big_blind + min_bet):
            return fold
        elif(prediction > (big_blind + min_bet) and prediction < (big_blind + min_bet)):
            return call/check
        else
            factor = prediction//big_blind
            return raise factor*big_blind

        """
    
    def play_step(self, game, final_move):
        return reward, done, score

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game()

    hand_memory = deque(maxlen=MAX_MEMORY)

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = agent.play_step(game, final_move)
        state_new = agent.get_state(game)

        # hand memory remember
        agent.short_remember(state_old, final_move, reward, state_new, done)

        # finished hand
        if done == 1:

            # train short mem
            while len(hand_memory) > 0:
                current_memory = hand_memory.popleft()
                agent.train_short_memory(current_memory[0], current_memory[1], current_memory[2], current_memory[3], current_memory[4])
                agent.remember(current_memory[0], current_memory[1], current_memory[2], current_memory[3], current_memory[4])

            agent.n_hands += 1

        # ran out of chips or too many hands
        elif done == 2:
            # train long mem
            agent.n_games += 1
            game.reset()
            if score > record:
                record = score
                # agent.mode.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            
            # plot


if __name__ == '__main__':
    train()
