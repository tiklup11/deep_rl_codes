# %%capture
# %%bash
# apt-get install cmake
# apt-get install zlib1g-dev
# pip install gym[atari]
# pip install JSAnimation

import numpy as np
# import cPickle as pickle
import matplotlib.pyplot as plt
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
import gym

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.optimizers import rmsprop
import keras.backend as K


%matplotlib inline

#animation of training
def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)

#test animations
# display(display_animation(anim, default_mode='once'))

observation = env.reset()
cum_reward = 0
frames = []
r = []
for t in range(100):
    # Render into buffer. 
    frames.append(env.render(mode = 'rgb_array'))
    p = np.random.dirichlet([1]*len(action_space), 1).ravel()
    a = np.random.choice(len(action_space), p=p)
    action = action_space[a]
    observation, reward, done, info = env.step(action)
    r.append(reward)
    if done:
        break
        
r = np.array(r)
# env.render(close=True)
# display_frames_as_gif(frames)
# print(t)


gamma = 0.99
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(len(discounted_r))):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add =  r[t] + running_add * gamma # belman equation
        discounted_r[t] = running_add
    return discounted_r

def discount_n_standardise(r):
    dr = discount_rewards(r)
    dr = (dr - dr.mean()) / dr.std()
    return dr


def preprocess(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float)[:,:,None]

#creating NN model
model = Sequential()
model.add(Conv2D(4, kernel_size=(3,3), padding='same', activation='relu', input_shape = (80,80,1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(8, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(12, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(len(action_space), activation='softmax'))
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy') #

model.summary()


episodes = 0
n_episodes = 1000
reward_sums = np.zeros(n_episodes)
losses = np.zeros(n_episodes)
time_taken = np.zeros(n_episodes)
reward_sum = 0
im_shape = (80, 80, 1)

prev_frame = None
buffer = 1000
xs = np.zeros((buffer,)+im_shape)
ys = np.zeros((buffer,1))
rs = np.zeros((buffer))

k = 0

observation = env.reset()


## training our NN
while episodes < n_episodes:
    # Get the current state of environment
    x = preprocess(observation)
    xs[k] = x - prev_frame if prev_frame is not None else np.zeros(im_shape)
    prev_frame = x
    
    # Take an action given current state of policy model
    p = model.predict(xs[k][None,:,:,:]) #return probablity distribution
    a = np.random.choice(len(action_space), p=p[0])
    action = action_space[a]
    ys[k] = a
    
    # Renew state of environment
    observation, reward, done, _ = env.step(action)
    reward_sum += reward #record total rewards
    rs[k] = reward # record reward per step
    
    k += 1
    
    if done or k==buffer:
        reward_sums[episodes] = reward_sum
        reward_sum = 0
        
        # Gather state, action (y), and rewards (and preprocess)
        ep_x = xs[:k]
        ep_y = ys[:k]
        ep_r = rs[:k]
        ep_r = discount_n_standardise(ep_r)
        
        model.fit(ep_x, ep_y, sample_weight=ep_r, batch_size=512, epochs=1, verbose=0)
        
        time_taken[episodes] = k
        k = 0
        prev_frame = None
        observation = env.reset()
        losses[episodes] = model.evaluate(ep_x, 
                                          ep_y,
                                          sample_weight=ep_r,
                                          batch_size=len(ep_x), 
                                          verbose=0)
        episodes += 1
        
        # Print out metrics like rewards, how long each episode lasted etc.
        if episodes%(n_episodes//20) == 0:
            ave_reward = np.mean(reward_sums[max(0,episodes-200):episodes])
            ave_loss = np.mean(losses[max(0,episodes-200):episodes])
            ave_time = np.mean(time_taken[max(0,episodes-200):episodes])
            print('Episode: {0:d}, Average Loss: {1:.4f}, Average Reward: {2:.4f}, Average steps: {3:.4f}'
                  .format(episodes, ave_loss, ave_reward, ave_time))
            

def plot():
    window = 20
    plt.plot(losses[:episodes])
    plt.plot(np.convolve(losses[:episodes], np.ones((window,))/window, mode='valid'))
    plt.show()

    plt.plot(reward_sums[:episodes])
    plt.plot(np.convolve(reward_sums[:episodes], np.ones((window,))/window, mode='valid'))
    plt.show()
    

observation = env.reset()
cum_reward = 0
frames = []
prev_frame = None
for t in range(1000):
    x = preprocess(observation) 
    diff = x - prev_frame if prev_frame is not None else np.zeros(im_shape)
    p = model.predict(diff[None,:,:,:])
    prev_frame = x
    a = np.random.choice(len(action_space), p=p[0])
    action = action_space[a]
    
    # Render into buffer. 
    frames.append(env.render(mode = 'rgb_array'))
    observation, reward, done, info = env.step(action)
    if done:
        break
        
# env.render(close=True)
display_frames_as_gif(frames)
# print(t)
    