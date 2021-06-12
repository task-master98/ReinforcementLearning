"""
@Author: Ishaan Roy
File contains: Simple Grid Environment
TODO:
--------------------------------------
=> Add environment for simple grid:
   -> Fixed dimensions of grid
   -> Grid contains portals (one cell -> another cell = 0 cost)
=> Simple Q-Learning: finite state, finite action
--------------------------------------
ChangeLog
--------------------------------------

--------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class GridWorld:
    """
    Terminology
    --------------------------------------
    State space: Set of all states excluding the terminal state
    Q-Table: A record of the states and actions with the corrseponding value function
    Grid Representation: {Empty squares: 0, Agent Position: 1, magic square: 1}
    --------------------------------------
    """
    def __init__(self, rows, cols, magicSquares):
        self.grid = np.zeros((rows, cols))
        self.rows = rows
        self.cols = cols
        self.stateSpace = [i for i in range(rows*cols)]
        self.stateSpace.remove(rows*cols - 1)
        self.stateSpacePlus = [i for i in range(rows*cols)]
        self.actionSpace = {
                            'U': -rows,
                            'D': rows,
                            'L': -1,
                            'R': +1
        }
        self.possibleActions = [key for key in self.actionSpace]
        self.addMagicSquares(magicSquares)
        self.agentPosition = 0

    def addMagicSquares(self, magicSquares):
        self.magicSquares = magicSquares
        magicSquare_identifier = 2
        for square in magicSquares:
            # src coordinates
            x = square // self.rows
            y = square % self.cols
            self.grid[x][y] = magicSquare_identifier
            magicSquare_identifier += 1
            # dst coordinates
            x = magicSquares[square] // self.rows
            y = magicSquares[square] % self.cols
            self.grid[x][y] = magicSquare_identifier
            magicSquare_identifier += 1

    def isTerminalState(self, state):
        return state in self.stateSpacePlus and state not in self.stateSpace

    def getAgentPosition(self):
        x = self.agentPosition // self.rows
        y = self.agentPosition % self.cols
        return (x, y)

    def setState(self, newState):
        x, y = self.getAgentPosition()
        self.grid[x][y] = 0
        self.agentPosition = newState
        x, y = self.getAgentPosition()
        self.grid[x][y] = 1

    def OffGridMove(self, newState, oldState):
        if newState not in self.stateSpacePlus:
            return True
        elif oldState % self.rows == 0 and newState % self.rows == self.rows - 1:
            return True
        elif oldState % self.rows == self.rows - 1 and newState % self.rows == 0:
            return True
        else:
            return False

    def step(self, action):
        x, y = self.getAgentPosition()
        resultingState = self.agentPosition + self.actionSpace[action]
        if resultingState in self.magicSquares.keys():
            resultingState = self.magicSquares[resultingState]

        reward = -1 if not self.isTerminalState(resultingState) else 0
        if not self.OffGridMove(resultingState, self.agentPosition):
            self.setState(resultingState)

            return resultingState, reward, self.isTerminalState(self.agentPosition), None
        else:
            return self.agentPosition, reward, self.isTerminalState(self.agentPosition), None

    def reset(self):
        self.agentPosition = 0
        self.grid = np.zeros((self.rows, self.cols))
        self.addMagicSquares(self.magicSquares)
        return self.agentPosition

    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)



    def render(self):
        print('----------------------------------------')
        for row in self.grid:
            for col in row:
                if col == 0:
                    print('-', end='\t')
                elif col == 1:
                    print('X', end='\t')
                elif col == 2:
                    print('Ain', end='\t')
                elif col == 3:
                    print('Aout', end='\t')
                elif col == 4:
                    print('Bin', end='\t')
                elif col == 5:
                    print('Bout', end='\t')
            print('\n')
        print('----------------------------------------')


def maxAction(Q, state, actions):
    values = np.array(Q[state, a] for a in actions)
    action = np.argmax(values)
    return actions[action]


if __name__ == "__main__":
    magicSquares = {18: 54, 63: 14}
    env = GridWorld(9, 9, magicSquares)
    env.render()

    ALPHA = 0.1
    GAMMA = 1.0
    EPS = 1.0

    Q = {}
    for state in env.stateSpacePlus:
        for action in env.possibleActions:
            Q[state, action] = 0

    NUMGAMES = 10
    totalRewards = np.zeros(NUMGAMES)
    for i in tqdm(range(NUMGAMES)):
        if i % 5000 == 0:
            print('Starting Game ', i)

        done = False
        epRewards = 0
        observation = env.reset()

        while not done:
            rand = np.random.random()
            if rand < (1 - EPS):
                action = maxAction(Q, observation, env.possibleActions)
            else:
                action = env.actionSpaceSample()

            observation_, reward, done, info = env.step(action)
            epRewards += reward
            action_ = maxAction(Q, observation_, env.possibleActions)
            Q[observation, action] = Q[observation, action] + ALPHA*(reward +
                                                                     GAMMA*Q[observation_, action_]
                                                                     - Q[observation, action])
            observation = observation_

        if EPS - 2 / NUMGAMES > 0:
            EPS -= 2/NUMGAMES
        else:
            EPS = 0

        totalRewards[i] = epRewards

    plt.plot(totalRewards)
    plt.show()





