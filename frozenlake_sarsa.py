import numpy as np
import gym
import time


class SARSA_algo:

    # env - Frozen Lake environment
    # alpha - step size
    # gamma - discount rate
    # epsilon - parameter for epsilon-greedy approach
    # numberEpisodes - total number of simulation episodes
    def __init__(self, env, alpha, gamma, epsilon, numberEpisodes):

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.stateNumber = env.observation_space.n
        self.actionNumber = env.action_space.n
        self.numberEpisodes = numberEpisodes
        # this vector is the learned policy
        self.learnedPolicy = np.zeros(env.observation_space.n)
        # this matrix is the action value function matrix
        # its entries are (s,a), where s is the state number and action is the action number
        # s=0,1,2,\ldots,15, a=0,1,2,3
        self.Qmatrix = np.zeros((self.stateNumber, self.actionNumber))

    # this function selects an action on the basis of the current state
    # INPUTS:
    # state - state for which to compute the action
    # index - index of the current episode

    def selectAction(self, state, try_counter):

        # first 100 episodes we select completely random actions to avoid being stuck
        if try_counter < 1000:
            return np.random.choice(self.actionNumber)

        # Returns a random real number in the half-open interval [0.0, 1.0)
        randomNumber = np.random.random()

        if try_counter > 4000:
            self.epsilon = 0.8*self.epsilon

        # if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber < self.epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.actionNumber)

        # otherwise, we are selecting greedy actions
        else:
            # we return the index where actionValueMatrixEstimate[state,:] has the max value
            # random because multiple state,action pair can have same Q-value
            return np.random.choice(np.where(self.Qmatrix[state, :] == np.max(self.Qmatrix[state, :]))[0])
            # here we need to return the minimum index since it can happen
            # that there are several identical maximal entries, for example
            # import numpy as np
            # a=[0,1,1,0]
            # np.where(a==np.max(a))
            # this will return [1,2], but we only need a single index
            # that is why we need to have np.random.choice(np.where(a==np.max(a))[0])
            # note that zero has to be added here since np.where() returns a tuple

    def simulateEpisodes(self):
        print("simulating episodes..")
        # here we loop through the episodes
        for indexEpisode in range(self.numberEpisodes):

            # reset the environment at the beginning of every episode
            
            (initial_state, prob) = self.env.reset()
            # select an action on the basis of the initial state
            action = self.selectAction(initial_state, indexEpisode)

            print("Simulating episode {}".format(indexEpisode))

            # here we step from one state to another
            # this will loop until a terminal state is reached
            done = False
            while not done:

                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                # prime means that it is the next state
                (next_state, reward, done, _, _) = self.env.step(action)

                # next action
                action = self.selectAction(next_state, indexEpisode)

                if not done:
                    error = reward+self.gamma * \
                        self.Qmatrix[next_state, action] - \
                        self.Qmatrix[initial_state, action]
                    self.Qmatrix[initial_state,
                                 action] = self.Qmatrix[initial_state, action]+self.alpha*error
                else:
                    # in the terminal state, we have Qmatrix[stateSprime,actionAprime]=0
                    error = reward-self.Qmatrix[initial_state, action]
                    self.Qmatrix[initial_state,
                                 action] = self.Qmatrix[initial_state, action]+self.alpha*error

                initial_state = next_state
                action = action

    def computeFinalPolicy(self):

        # now we compute the final learned policy
        for index in range(self.stateNumber):
            # we use np.random.choice() because in theory, we might have several identical maximums
            self.learnedPolicy[index] = np.random.choice(
                np.where(self.Qmatrix[index] == np.max(self.Qmatrix[index]))[0])


#########################
if __name__ == "__main__":

    env = gym.make('FrozenLake-v1', desc=None,
                   map_name="4x4", is_slippery=False,render_mode="human")
    env.reset()
    env.render()
    
    # step size
    alpha = 0.1
    # discount rate
    gamma = 0.9
    # epsilon-greedy parameter
    epsilon = 0.2
    # number of simulation episodes
    numberEpisodes = 25000

    # initialize
    SARSA1 = SARSA_algo(env, alpha, gamma, epsilon, numberEpisodes)
    # simulate
    SARSA1.simulateEpisodes()
    # compute the final policy
    SARSA1.computeFinalPolicy()

    # extract the final policy
    finalLearnedPolicy = SARSA1.learnedPolicy

    # simulate the learned policy for verification
    while True:
        print("playing for real")
        # to interpret the final learned policy you need this information
        # actions: 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
        # let us simulate the learned policy
        # this will reset the environment and return the agent to the initial state
        env = gym.make('FrozenLake-v1', desc=None, map_name="4x4",
                       is_slippery=False, render_mode='human')
        (currentState, prob) = env.reset()
        env.render()
        time.sleep(2)
        # since the initial state is not a terminal state, set this flag to false
        terminalState = False
        for i in range(100):
            # here we step and return the state, reward, and boolean denoting if the state is a terminal state
            if not terminalState:
                (currentState, currentReward, terminalState, _, _) = env.step(
                    int(finalLearnedPolicy[currentState]))
                time.sleep(1)
            else:
                break
        time.sleep(0.5)
    env.close()
