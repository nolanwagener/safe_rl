Safe Reinforcement Learning Using Advantage-Based Intervention
==============================================================

Source code for [Safe Reinforcement Learning Using Advantage-Based Intervention](https://arxiv.org/abs/2106.09110).
Code based on [OpenAI's Spinning Up](https://spinningup.openai.com/).

Installation Instructions
-------------------------
1. Except for the notes below, follow the installation instructions here: https://spinningup.openai.com/en/latest/user/installation.html
    - Replace `conda create -n spinningup python=3.6` and `conda activate spinningup` with `conda create -n safe python=3.6` and `conda activate safe`, respectively.
	- Instead of the "Installing Spinning Up" section, run the following:
	```
	git clone https://github.com/nolanwagener/safe_rl.git
	cd safe_rl
	pip install -e .
	```
	- Also follow the instructions in "Installing MuJoCo".
2. Go to the `extra_envs` directory and install it to expose those environments to Gym.
```
cd extra_envs
pip install -e .
```
3. Install the [`mjmpc`](https://github.com/mohakbhardwaj/mjmpc/tree/nolan/safe_rl) and [`mjrl`](https://github.com/aravindr93/mjrl) repos for the MPC-based Half-Cheetah intervention.
```
git clone https://github.com/mohakbhardwaj/mjmpc.git -b nolan/safe_rl
pip install git+https://github.com/aravindr93/mjrl.git@master#egg=mjrl
cd mjmpc
pip install -e .
```

Code Architecture
-----------------
- `extra_envs`: This consists of three folders:
	- `envs`: The point and Half-Cheetah environments.
	- `wrappers`: Defines the intervention wrapper. Whenever we `step` the wrapped environment, we check whether the agent should be intervened. If not, we `step` the internal environment. Otherwise, we return a NaN observation and set the `done` flag to `True`. If the intervener gives a safe action, the returned `info` dictionary includes the `step` output `(o, r, d, info)` when the safe action from the intervener is applied to the internal environment.
	- `intervener`: Intervention rules G = (Q, μ, η) for the point and Half-Cheetah environments. The rule contains a `should_intervene` method which uses a heuristic or an advantage-based rule to decide whether a given action requires intervention. The `safe_action` method returns a safe action from μ.
		- Point: The safe policy μ is a deceleration policy. We have two interveners:
			- `PointIntervenerNetwork`: Uses value and Q-value approximators to build the advantage function estimate. The networks are loaded using PyTorch.
			- `PointIntervenerRollout`: Rolls out the deceleration policy on a model of the environment to build the advantage function estimate.
		- Half-Cheetah:
		    - `HalfCheetahHeuristicIntervener`: Merely checks if the predicted next state would result in a constraint violation. The episode immediately terminates upon intervention.
			- `HalfCheetahMpcIntervener`: Uses a modeled environment and sampling-based MPC to form the safe policy. Similarly builds an advantage-function estimate using MPC. Upon intervention, can either reset the environment or return an action from MPC.
- `safe_rl/algos`: 
	- `cppo`: The PPO algorithm modified for the constrained/safe setting. Our implementation maintains a value function for the reward and a value function to predict the constraint cost or an intervention (here, overloaded into the same scalar). Thus, CPPO can be used for both the constrained setting where a Lagrange multiplier is optimized and the unconstrained safe setting where we receive a fixed penalty for an intervention.
	- `csc`: The constrained PPO algorithm but with a state-action critic used for the cost in place of the state critic. The state-action "safety" critic is used to filter out unsafe proposed actions, and is trained in a conservative fashion to make the agent more safe.


Scripts
-------
All training scripts are located in `/scripts/` within two 'script.sh' files. After running the scripts, the results are stored in `/data`.
Results can be plotted using the "Spinning Up" plotting tool (with `spinup.run` replaced with `safe_rl.run`): https://spinningup.openai.com/en/latest/user/plotting.html

Citing
------
If you use this repo in your research, please cite:
```
@inproceedings{wagener2021safe,
  title={{Safe Reinforcement Learning Using Advantage-Based Intervention}},
  author={Wagener, Nolan and Boots, Byron and Cheng, Ching-An},
  booktitle={International Conference on Machine Learning},
  pages={10630--10640},
  year={2021},
  organization={PMLR}
}
```
