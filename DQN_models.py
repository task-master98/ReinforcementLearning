"""
@Author: Ishaan Roy
#####################
Deep Q-Learning Algorithm
Framework: Pytorch
Task: Using Reinforcement Learning to play Space Invaders
File contains: Models required for Space Invaders game

#TODO
----------------------------------
=> Implement the Agent class: same file?
=> Write the main loop
=> Move helper function to utils.py
=> Add comments to explain the action space
----------------------------------
Change Log
-----------------------------------
=> Initial Commit: Built the model
    -> Deep Convolution network: model tested
=> Implementation of RL agent
    -> RL agent not tested yet; to be tested in the main loop
=> Helper function to plot rewards written

-----------------------------------
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class DeepQNetwork(nn.Module):
    def __init__(self, lr, ):
        super(DeepQNetwork, self).__init__()
        self.lr = lr
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=1)
        self.conv_2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv_3 = nn.Conv2d(64, 128, 3)
        self.conv = nn.Sequential(
                    self.conv_1,
                    nn.ReLU(),
                    self.conv_2,
                    nn.ReLU(),
                    self.conv_3,
                    nn.ReLU(),
                    nn.Flatten()
        )

        self.fc_1 = nn.Linear(128*19*8, 512)
        self.fc_2 = nn.Linear(512, 64)
        self.fc_3 = nn.Linear(64, 6)
        self.linear = nn.Sequential(
                        self.fc_1,
                        nn.ReLU(),
                        self.fc_2,
                        self.fc_3
        )


        self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, observation):
        observation = torch.Tensor(observation).to(self.device)
        observation = observation.view(-1, 1, 185, 95)
        feature_map = self.conv(observation)
        actions = self.linear(feature_map)
        return actions


class Agent:
    def __init__(self, gamma, epsilon, lr,
                 maxMemsize, epsEnd=0.05,
                 replace=10000, action_space=list(range(6))):
        ## Parameters of Policy
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.actionSpace = action_space
        self.maxMemsize = maxMemsize
        self.replace_target_cnt = replace

        ## RL tools
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0

        ## Models
        self.Q_eval = DeepQNetwork(lr=lr)
        self.Q_target = DeepQNetwork(lr=lr)

    def storeTransitions(self, old_state, action,
                         reward, new_state):
        if self.memCntr < self.maxMemsize:
            self.memory.append([old_state, action, reward, new_state])
        else:
            self.memory[self.memCntr%self.maxMemsize] = [old_state, action, reward, new_state]

        self.memCntr += 1

    def chooseAction(self, observation):
        np.random.seed(42)
        rand = np.random.random()
        actions = self.Q_eval(observation)
        if rand < 1 - self.EPSILON:
            action = torch.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.actionSpace)
        self.steps += 1
        return action

    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()
        if (self.replace_target_cnt is not None and
            self.learn_step_counter % self.replace_target_cnt == 0):
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        ## Sampling from the memory bank (randomly)
        if self.memCntr + batch_size < self.maxMemsize:
            memStart = int(np.random.choice(range(self.memCntr)))
        else:
            memStart = int(np.random.choice(range(self.memCntr-batch_size-1)))
        miniBatch = self.memory[memStart:memStart+batch_size]
        memory = np.array(miniBatch)

        Qpred = self.Q_eval.forward(list(memory[:, 0])).to(self.Q_eval.device)
        Qnext = self.Q_target.forward(list(memory[:, 3])).to(self.Q_eval.device)

        maxA = torch.argmax(Qnext, dim=1).to(self.Q_eval.device)
        rewards = torch.Tensor(list(memory[:, 2])).to(self.Q_eval.device)
        Qtargets = Qpred
        Qtargets[:, maxA] = rewards + self.GAMMA*torch.max(Qnext[1])

        if self.steps > 500:
            if self.EPSILON - 1e-4 < self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END

        loss = self.Q_eval.loss(Qtargets, Qpred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

def plotLearning(x, scores, epsilons, filename=None, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    return fig









def test_model():
    img = torch.randn((185, 95))
    model = DeepQNetwork(0.003)
    print(model.forward(img).shape)

if __name__ == "__main__":
    test_model()



