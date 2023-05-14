import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE=128
BATCH_SIZE=16
PERCENTILE=70

#Net is our neural network
class Net(nn.Module):
    def __init__(self,obs_size,hidden_size,n_actions):
        super(Net,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,n_actions)
        )
    def forward(self,x):
        return self.net(x)
    

#helper classes or custom classes
Episode = namedtuple('Episode',field_names=['reward','steps'])
EpisodeStep = namedtuple('EpisodeStep',field_names=['observation','action'])

def iterate_batches(env,net,batch_size):
    batch = [] #List of episodes
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    #used to convert the network output to a probabilty distribution
    #dim parameter specifies the dimension along which the softmax should be applied
    #dim =1 second dimention
    sm = nn.Softmax(dim=1)

    while True:
        print(obs)
        obs_t = torch.FloatTensor([obs])
        # obs_t = torch.from_numpy(np.array(obs)).float()
        # obs_t = torch.from_numpy(np.stack(obs)).float()
        # obs_t = obs_t.view(1, -1)  
        action_props_t = sm(net(obs_t)) #net expects a batch of items
        action_props = action_props_t.data.numpy()[0]

        action = np.random.choice(len(action_props),p = action_props)
        next_obs, reward, is_done, _, _ = env.step(action)

        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs,action=action))

        if is_done:
            batch.append(Episode(reward=episode_reward,steps=episode_steps))
            episode_steps = []
            episode_reward = 0.0
            next_obs = env.reset()
            if len(batch)==batch_size:
                yield batch
                batch = []
        obs = next_obs

def filter_batch(batch,percentile):
    print("filtering the batch")
    rewards = list(map(lambda s:s.reward,batch)) #mapes into list of rewards
    print("rewards list : ", rewards)
    reward_bound = np.percentile(rewards,percentile)
    print("rewards_bound : ",reward_bound)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_acts = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step:step.observation,example.steps))
        train_acts.extend(map(lambda step:step.action,example.steps))

    train_obs_t = torch.FloatTensor(train_obs)
    train_acts_t = torch.FloatTensor(train_acts)
    return train_obs_t, train_acts_t, reward_bound, reward_mean

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode = "human")
    
    obs_size = env.observation_space.shape[0]
    print("obs space : ")
    print(env.observation_space)
    n_actions = env.action_space.n
    print("action space")
    print(env.action_space)
    
    net = Net(obs_size,HIDDEN_SIZE,n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(),lr=0.01)
    writer = SummaryWriter()
    
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_t, acts_t, reward_b, reward_m = filter_batch(batch,PERCENTILE)
        optimizer.zero_grad()
        action_scores_t = net(obs_t)
        loss_t = objective(action_scores_t,acts_t)
        loss_t.backwards() #Points to parents of the graph
        optimizer.step() #calculates new values of weights and baises in the direction of gradient
        
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_t.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_t.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break
        
        writer.close()








