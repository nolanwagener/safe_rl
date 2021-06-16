import argparse
import numpy as np
import scipy.signal
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--cost_smoothing', type=float, default=0)
args = parser.parse_args()

cost_smoothing = args.cost_smoothing
cost_fn = ((lambda d: np.array(d == 0., dtype=np.float32))
           if cost_smoothing == 0.
           else (lambda d: np.maximum(0, 1-d/cost_smoothing)))
gamma = 0.99

def discounted_return(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[-1]

q_dists = np.load('npz/q_dists.npz', allow_pickle=True)
v_dists = np.load('npz/v_dists.npz', allow_pickle=True)

# Q-function
s, a, dists = [q_dists[k] for k in ('s', 'a', 'dists')]
q = []
for i in tqdm(range(s.shape[0])):
    d = dists[i]
    ret = discounted_return(cost_fn(d), gamma)
    if d[-1] == 0.:
        ret += gamma**(d.shape[0]+1) / (1-gamma)
    q.append(ret)
np.savez('npz/q_smooth_{}.npz'.format(cost_smoothing), s=s, a=a, q=np.array(q))

# Value function
s, dists = [v_dists[k] for k in ('s', 'dists')]
v = []
for i in tqdm(range(s.shape[0])):
    d = dists[i]
    ret = discounted_return(cost_fn(d), gamma)
    if d[-1] == 0.:
        ret += gamma**(d.shape[0]+1) / (1-gamma)
    v.append(ret)

np.savez('npz/v_smooth_{}.npz'.format(cost_smoothing), s=s, v=np.array(v))
