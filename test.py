import numpy as np
# n_steps = np.array([1,2,3443])
from collections import deque

n_steps = 10
returns = deque(maxlen=n_steps)
gamma = 0.9
rewards = [1,0,1,0,1,0,1,0,0,10]

print(f"returns deque look like = {returns}")
for t in range(n_steps)[::-1]:
    print(f"at step : {t}, reward = {rewards[t]}")
    disc_return_t = returns[0] if len(returns) > 0 else 0
    
    return_at_t = gamma * disc_return_t + rewards[t]
    print(f"discounted return at {t} => {return_at_t}")
    returns.appendleft(gamma * disc_return_t + rewards[t])
    print(f"returns deque look like = {returns}")
    
    
    