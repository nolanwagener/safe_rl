import numpy as np
import torch
import torch.nn as nn
import scipy.signal
from .base import Intervener

def discounted_return(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[-1]

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class PointIntervener(Intervener):
    """
    Intervener for the point system. Intervenes by selecting a safe action based on an
    internal model of the point system and environment.
    """
    def __init__(self, sim_env, mode=Intervener.MODE.SAFE_ACTION,
                 cost_smoothing=0., gamma=0.99, intv_lim=0.01, only_q=False,
                 **kwargs):
        super().__init__(mode)

        self.sim_env = sim_env.unwrapped
        self.cost_fn = ((lambda d: np.array(d == 0., dtype=np.float32))
                        if cost_smoothing == 0.
                        else lambda d: np.maximum(0, 1 - d/cost_smoothing))
        self.gamma = gamma
        self.intv_lim = intv_lim
        self.only_q = only_q

    def reset(self, **kwargs):
        self.sim_env.reset()

    def set_state(self, env_state):
        self.sim_env.state = env_state
        self.state = np.array(env_state)

    def safe_action(self):
        """
        An action to cause the velocity to be zero, up to the maximum action magnitude.
        """
        vel = self.sim_env.state[2:]
        low = float(self.sim_env.action_space.low[0])
        high = float(self.sim_env.action_space.high[0])
        a = np.clip(-self.sim_env.mass*vel/self.sim_env.dt, low, high)
        return np.array(a, dtype=np.float32)

class PointIntervenerRollout(PointIntervener):
    """
    Intervention rule uses a model-based rollout.
    """
    def should_intervene(self, action):
        distances = self.safe_pi_distances(action)
        costs = self.cost_fn(distances)
        qc = discounted_return(costs, self.gamma)

        if self.only_q:
            return qc > self.intv_lim

        opt_distances = self.safe_pi_distances(self.safe_action())
        opt_costs = self.cost_fn(opt_distances)
        vc = discounted_return(opt_costs, self.gamma)

        disadv = qc - vc
        return disadv > self.intv_lim

    def safe_pi_distances(self, act):
        start_state = np.array(self.sim_env.state)
        distances = [self.sim_env.dist_to_unsafe()]
        while True:
            _, _, _, info = self.sim_env.step(act)
            distances.append(info['distance'])
            act = self.safe_action()
            if (self.sim_env.state[2:] == 0.).all() or info['distance'] == 0.:
                break

        self.sim_env.state = start_state
        return np.array(distances)

class PointIntervenerNetwork(PointIntervener):
    """
    Intervention rule uses neural networks.
    """
    def __init__(self, sim_env, vnet1_path, vnet2_path, qnet1_path, qnet2_path,
                 mode=Intervener.MODE.SAFE_ACTION, intv_lim=0.01, only_q=False,
                 **kwargs):
        super().__init__(sim_env, mode=mode, intv_lim=intv_lim, only_q=only_q, **kwargs)
        self.qnet1 = mlp([4+2, 256, 256, 256, 1], nn.ReLU)
        self.qnet1.load_state_dict(torch.load(qnet1_path))
        self.qnet1.eval()
        self.qnet1.cpu()
        self.qnet2 = mlp([4+2, 256, 256, 256, 1], nn.ReLU)
        self.qnet2.load_state_dict(torch.load(qnet2_path))
        self.qnet2.eval()
        self.qnet2.cpu()

        self.vnet1 = mlp([4, 256, 256, 256, 1], nn.ReLU)
        self.vnet1.load_state_dict(torch.load(vnet1_path))
        self.vnet1.eval()
        self.vnet1.cpu()
        self.vnet2 = mlp([4, 256, 256, 256, 1], nn.ReLU)
        self.vnet2.load_state_dict(torch.load(vnet2_path))
        self.vnet2.eval()
        self.vnet2.cpu()

    def should_intervene(self, action):
        q_inp = torch.from_numpy(np.concatenate([self.state, action])).unsqueeze(0)
        q1 = torch.sigmoid(self.qnet1(q_inp.float())).squeeze().item()
        q2 = torch.sigmoid(self.qnet2(q_inp.float())).squeeze().item()
        q = max(q1, q2)

        if self.only_q:
            return q > self.intv_lim

        v_inp = torch.from_numpy(self.state).unsqueeze(0)
        v1 = torch.sigmoid(self.vnet1(v_inp.float())).squeeze().item()
        v2 = torch.sigmoid(self.vnet2(v_inp.float())).squeeze().item()
        v = min(v1, v2)

        return q-v > self.intv_lim
