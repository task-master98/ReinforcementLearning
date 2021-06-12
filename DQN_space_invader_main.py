"""
@Author: Ishaan Roy
Deep Q-Learning Algorithm
Framework: Pytorch
Task: Using Reinforcement Learning to play Space Invaders
File contains: Main loop for training the agent

#TODO
----------------------------------
=> Fix issue: IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
              in line 130 in DQN_models.py
=> Reduce batch size in case code keeps crashing
=> Run code on google collab

----------------------------------

Change Log
----------------------------------
=> Initial commit
    -> Added main loop for training the RL agent
    -> Added environment renderer

----------------------------------
"""
import gym
try:
    from .DQN_models import Agent, plotLearning
except ImportError:
    from DQN_models import Agent, plotLearning
import numpy as np

if __name__ == "__main__":
    env = gym.make('SpaceInvaders-v0')
    agent = Agent(gamma=0.99, epsilon=1.0,
                  lr=0.003, maxMemsize=5000,
                  replace=None)

    while agent.memCntr < agent.maxMemsize:
        observation = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            if done and info['ale.lives'] == 0:
                reward = -100
            agent.storeTransitions(np.mean(observation[15:200, 30:125], axis=2), action,
                                   reward, np.mean(observation_[15:200, 30:125], axis=2))
            observation = observation_
    print('Done Initialising Memory')

    scores = []
    epsHistory = []
    numGames = 50
    batch_size = 32

    for i in range(numGames):
        print('Staring Game: ', i+1, 'epsilon: ', agent.EPSILON)
        epsHistory.append(agent.EPSILON)
        done = False
        observation = env.reset()
        frames = [np.sum(observation[15:200, 30:125], axis=2)]
        score = 0
        lastAction = 0

        while not done:
            if len(frames) == 3:
                action = agent.chooseAction(frames)
                frames = []
            else:
                action = lastAction

            observation_, reward, done, info = env.step(action)
            score += reward
            frames.append(np.sum(observation_[15:200, 30:125], axis=2))
            if done and info['ale.lives'] == 0:
                reward = -100
            agent.storeTransitions(np.mean(observation[15:200, 30:125], axis=2), action,
                                   reward, np.mean(observation_[15:200, 30:125], axis=2))
            observation = observation_
            agent.learn(batch_size)
            lastAction = action
            env.render()
        scores.append(score)
        print('Score: ', score)
        x = [i+1 for i in range(numGames)]
        fig = plotLearning(x, scores, epsHistory)
        fig.show()



