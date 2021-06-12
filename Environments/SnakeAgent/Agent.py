"""
@Author: Ishaan Roy
File contain: RL agent for snake environment
TODO:
------------------------------------------
=> Store model and trainer in agent
------------------------------------------
ChangeLog
------------------------------------------

------------------------------------------
"""
import numpy as np
import random
import torch
from collections import deque
try:
    from .Snake import SnakeGameAI, Direction, Point
    from .Brain import Brain, QTrainer
    from .Utils import plot
except ImportError:
    from Snake import SnakeGameAI, Direction, Point
    from Brain import Brain, QTrainer
    from Utils import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Brain(11, 256, 3)
        self.trainer = QTrainer(self.model, alpha=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_d = Point(head.x, head.y + 20)
        point_u = Point(head.x, head.y - 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_d = game.direction == Direction.DOWN
        dir_u = game.direction == Direction.UP

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_u)),

            # Danger right
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_r)),

            # Danger Left
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)),

            # Current direction
            dir_r,
            dir_l,
            dir_d,
            dir_u,

            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y

        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff between exploration and exploitation
        self.epsilon = 80 - self.num_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    env = SnakeGameAI()

    while True:
        # Current state
        state = agent.get_state(env)
        # Action based on state
        final_move = agent.get_action(state)
        # Perform action to get observation
        reward, done, score = env.play_step(action=final_move)
        # get new state based on observation
        next_state = agent.get_state(env)

        # Train short memory
        agent.train_short_memory(state, final_move, reward, next_state, done)

        # Remember
        agent.remember(state, final_move, reward, next_state, done)

        if done:
            env.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record:
                record = score

            print('Game: ', agent.num_games, 'Score: ', score, 'Record: ', record)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()