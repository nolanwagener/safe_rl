#!/usr/bin/env python

import argparse
import yaml
import gym
from safe_rl import cppo
import safe_rl.algos.cppo.core as core
from safe_rl.utils.mpi_tools import mpi_fork
from safe_rl.utils.run_utils import setup_logger_kwargs
from extra_envs.wrappers import Intervention
from extra_envs.intervener import Intervener, HalfCheetahHeuristicIntervener, HalfCheetahMpcIntervener

parser = argparse.ArgumentParser()

# Neural network
parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--l', type=int, default=2)

# Sampling
parser.add_argument('--steps', type=int, default=4000)
parser.add_argument('--cpu', type=int, default=4)

# Optimization
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--ent_bonus', type=float, default=0.01)
parser.add_argument('--dont_normalize_adv', '-dna', action='store_true')

# Cost function
parser.add_argument('--cost_lim', type=float, default=0.01)
parser.add_argument('--penalty', type=float, default=30.)
parser.add_argument('--optimize_penalty', action='store_true')
parser.add_argument('--ignore_unsafe_cost', action='store_true')
parser.add_argument('--penalty_lr', type=float, default=5e-2)

# Intervention
parser.add_argument('--intv_config', type=str, default='',
                    help="Path to intervener config file. No intv if empty.")
parser.add_argument('--heuristic_intv', action='store_true')

# Misc.
parser.add_argument('--exp_name', type=str, default='cheetah')
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--cost_smoothing', type=float, default=0.)
parser.add_argument('--num_test_episodes', type=int, default=10)

args = parser.parse_args()

mpi_fork(args.cpu)
logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

if args.intv_config == '':
    do_intv = False
    intv_kwargs = dict()
else:
    do_intv = True
    with open(args.intv_config) as f:
        intv_kwargs = yaml.load(f, Loader=yaml.FullLoader)
        if intv_kwargs is None:
            intv_kwargs = dict()

def env_fn():
    env = gym.make('extra_envs:HalfCheetah-v0')
    if 'gamma' not in intv_kwargs:
        intv_kwargs['gamma'] = args.gamma
    if 'mode' in intv_kwargs:
        if intv_kwargs['mode'] == 'TERMINATE':
            intv_kwargs['mode'] = Intervener.MODE.TERMINATE
        elif intv_kwargs['mode'] == 'SAFE_ACTION':
            intv_kwargs['mode'] = Intervener.MODE.SAFE_ACTION
        else:
            raise ValueError(intv_kwargs['mode'])
    if do_intv:
        IntervenerCls = (HalfCheetahHeuristicIntervener
                         if args.heuristic_intv
                         else HalfCheetahMpcIntervener)
    else:
        IntervenerCls = Intervener
    intervener = IntervenerCls(**intv_kwargs)
    return Intervention(env, intervener)

cppo(env_fn, actor_critic=core.MLPActorCritic,
     ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, seed=args.seed,
     dont_normalize_adv=args.dont_normalize_adv, steps_per_epoch=args.steps,
     epochs=args.epochs, logger_kwargs=logger_kwargs, cost_lim=args.cost_lim,
     penalty=args.penalty, optimize_penalty=args.optimize_penalty,
     penalty_lr=args.penalty_lr, ent_bonus=args.ent_bonus,
     ignore_unsafe_cost=args.ignore_unsafe_cost, num_test_episodes=args.num_test_episodes)
