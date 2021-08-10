from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import gym
import time
import yaml
import safe_rl.algos.csc.core as core
from safe_rl.utils.logx import EpochLogger
from safe_rl.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from safe_rl.utils.mpi_tools import (mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar,
                                    num_procs, mpi_sum)
from extra_envs.wrappers import Intervention
from extra_envs.intervener.base import Intervener


class CPPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, scaling=1.,
                 normalize_adv=False):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)

        # Associated with task reward
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)

        # Associated with task cost
        self.cadv_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.cret_buf = np.zeros(size, dtype=np.float32)
        self.cval_buf = np.zeros(size, dtype=np.float32)

        self.intv_buf = np.zeros(size, dtype=np.bool)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.normalize_adv = normalize_adv
        self.gamma, self.lam, self.scaling = gamma, lam, scaling
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, cost, cval, intv, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act

        # Reward
        self.rew_buf[self.ptr] = self.scaling*rew
        self.val_buf[self.ptr] = val

        # Cost
        self.cost_buf[self.ptr] = self.scaling*cost
        self.cval_buf[self.ptr] = cval

        self.intv_buf[self.ptr] = intv
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0, last_cval=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)

        ###########
        # Rewards #
        ###########
        rews = np.append((1-self.gamma)*self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        #########
        # Costs #
        #########
        costs = np.append((1-self.gamma)*self.cost_buf[path_slice], last_cval)
        cvals = np.append(self.cval_buf[path_slice], last_cval)
        cdeltas = costs[:-1] + self.gamma*cvals[1:] - cvals[:-1]
        self.cadv_buf[path_slice] = core.discount_cumsum(cdeltas, self.gamma*self.lam)
        self.cret_buf[path_slice] = core.discount_cumsum(costs, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, log_penalty=-np.infty):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        weight = 1/(1 + np.exp(-log_penalty))
        lagrange_adv_buf = (1-weight)*self.adv_buf - weight*self.cadv_buf
        adv_mean, adv_std = mpi_statistics_scalar(lagrange_adv_buf)
        lagrange_adv_buf = lagrange_adv_buf - adv_mean
        lagrange_adv_buf /= (adv_std if self.normalize_adv else self.scaling)

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    cret=self.cret_buf, adv=lagrange_adv_buf, logp=self.logp_buf)
        out = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
        out.update(intv=self.intv_buf)
        return out

class CSCBuffer:
    def __init__(self, obs_dim, act_dim, size, scaling=1.):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.scaling = scaling
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, cost, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.cost_buf[self.ptr] = self.scaling*cost
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     cost=self.cost_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

def csc(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=1e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10,
        num_test_episodes=10, ent_bonus=0.001, scaling=100., dont_normalize_adv=False,
        # Cost constraint/penalties
        cost_lim=0.01, penalty=1., penalty_lr=5e-2, update_penalty_every=1,
        optimize_penalty=False, ignore_unsafe_cost=False,
        # CSC
        replay_size=int(1e6), batch_size=100, alpha=5., polyak=0.995
        ):
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        scaling (float): How much to scale the empirical returns to aid in learning the
            value function

        cost_lim (float): The tolerated cost limit

        penalty (float): The initial penalty value

        penalty_lr (float): The update size for the penalty

        update_penalty_every (int): After how many policy updates we update the penalty

        optimize_penalty (bool): Whether to optimize the penalty or keep it fixed

        ignore_unsafe_cost (bool): Whether to consider the unsafe cost when computing the
            cost.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    test_env = env.env
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    rew_range = env.reward_range
    v_range = (scaling*rew_range[0], scaling*rew_range[1])
    vc_range = (0, scaling*1)
    max_ep_len = min(max_ep_len, env.env._max_episode_steps)

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, v_range=v_range,
                      vc_range=vc_range, pred_std=True, eps=cost_lim, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = CPPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, scaling,
                     normalize_adv=not dont_normalize_adv)
    csc_buf = CSCBuffer(obs_dim, act_dim, replay_size, scaling)

    # Penalty learning
    if optimize_penalty:
        #penalty_param_init = max(np.log(penalty), -100.)
        penalty_param_init = max(np.log(np.exp(penalty)-1), -100.)
        penalty_param = torch.tensor([penalty_param_init], requires_grad=True,
                                     dtype=torch.float32)
        #penalty_torch = torch.exp(penalty_param)
        penalty_torch = F.softplus(penalty_param)
    else:
        penalty_torch = torch.tensor([penalty], dtype=torch.float32)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        intv = data['intv']
        obs, act, adv, logp_old = [x[~intv] for x in [obs, act, adv, logp_old]]

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        ent = pi.entropy().mean().item()
        loss_pi = -(torch.min(ratio * adv, clip_adv) + ent_bonus*ent).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    loss_fn = torch.nn.SmoothL1Loss()
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return loss_fn(ac.v(obs), ret)
    def compute_loss_penalty(cost):
        return -penalty_torch.squeeze()*(cost - cost_lim)

    def compute_loss_qc(data):
        o, a, c, o2, d = [data[s] for s in ('obs', 'act', 'cost', 'obs2', 'done')]
        q = ac.qc(o, a)
        with torch.no_grad():
            pi_next = ac_targ.pi._distribution(o2)
            #pi_next = ac.pi._distribution(o2)
            acts_next = pi_next.sample_n(100)
            obs_next_flat = o2.unsqueeze(0).repeat(100, 1, 1).view(-1, obs_dim[0])
            acts_next_flat = acts_next.view(-1, act_dim[0])
            q_pi_targ_flat = torch.clamp(ac_targ.qc(obs_next_flat, acts_next_flat),
                                         *vc_range)
            #q_pi_targ_flat = torch.clamp(ac.qc(obs_next_flat, acts_next_flat),
            #                             *vc_range)
            q_pi_targ = q_pi_targ_flat.view(100, -1).mean(0)
            backup = c + gamma*(1-d)*q_pi_targ

        # Conservative regularization
        with torch.no_grad():
            pi = ac.pi._distribution(o)
            acts = pi.sample_n(100)
            obs_flat = o.unsqueeze(0).repeat(100, 1, 1).view(-1, obs_dim[0])
            acts_flat = acts.view(-1, act_dim[0])
        q_pi = ac.qc(obs_flat, acts_flat).view(100, -1).mean(0)

        loss = loss_fn(q, backup) + alpha*(q-q_pi).mean()
        #loss = loss_fn(q, backup) - alpha*q_pi.mean()
        loss_info = dict(QcVals=q.detach().numpy())

        return loss, loss_info

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    qcf_optimizer = Adam(ac.qc.parameters(), lr=vf_lr)
    if optimize_penalty:
        penalty_optimizer = Adam([penalty_param], lr=penalty_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(update_penalty):
        curr_cost = logger.get_stats('EpCost')[0]
        if curr_cost > cost_lim:
            logger.log("Warning! Safety constraint violated.", 'red')
        data = buf.get(np.log(penalty_torch.item()))

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

            qcf_optimizer.zero_grad()
            csc_data = csc_buf.sample_batch(batch_size)
            loss_qc, loss_info = compute_loss_qc(csc_data)
            loss_qc.backward()
            mpi_avg_grads(ac.qc)
            qcf_optimizer.step()

        # Penalty update
        if optimize_penalty and update_penalty:
            penalty_optimizer.zero_grad()
            loss_pen = compute_loss_penalty(curr_cost)
            loss_pen.backward()
            penalty_param_np = penalty_param.grad.numpy()
            avg_grad = mpi_avg(penalty_param_np)
            penalty_param.grad = torch.as_tensor(avg_grad)
            penalty_optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)


        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old, LossQC=loss_qc,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old), **loss_info)

    local_num_test_episodes = int(num_test_episodes / num_procs())
    def test_agent():
        for _ in range(local_num_test_episodes):
            o, d, ep_ret, ep_cost, ep_len = test_env.reset(), False, 0, 0, 0
            while not (d or ep_len == max_ep_len):
                a = ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic=True)
                o, r, d, info = test_env.step(a)
                ep_ret += r
                ep_cost += info.get('cost', 0.)
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpCost=ep_cost, TestEpLen=ep_len)

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_cost, ep_len, cum_cost, cum_viol = env.reset(), 0, 0, 0, 0, 0
    ep_surr_cost, cum_surr_cost = 0, 0
    already_intv, local_episodes = False, 1

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        if optimize_penalty:
            #penalty_torch = torch.exp(penalty_param)
            penalty_torch = F.softplus(penalty_param)

        for t in range(local_steps_per_epoch):
            a, v, vc, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            next_o, r, d, info = env.step(a)

            intervened = info.get('intervened', False)
            if not intervened:
                c = info.get('cost', 0.)
                violation = (c == 1.)

                ep_ret += r
                ep_cost += c
                ep_len += 1
                cum_cost += c
                cum_viol += violation

                surr_c = (1-ignore_unsafe_cost)*c
                ep_surr_cost += surr_c
                cum_surr_cost += surr_c

                buf.store(o, a, r, v, 0., vc, already_intv, logp)
                csc_buf.store(o, a, c, next_o, d)
            elif env.intervener.mode == env.intervener.MODE.SAFE_ACTION:
                next_o, r_safe, d, info_safe = info['safe_step_output']
                c_safe = info_safe.get('cost', 0.)
                violation = (c_safe == 1.)

                ep_ret += r_safe
                ep_cost += c_safe
                ep_len += 1
                cum_cost += c_safe
                cum_viol += violation

                ep_surr_cost += 1.
                cum_surr_cost += 1.

                buf.store(o, a, 0.*r, v, 1., vc, already_intv, logp)
            elif env.intervener.mode == env.intervener.MODE.TERMINATE:
                violation = False
                ep_surr_cost += 1.
                cum_surr_cost += 1.
                buf.store(o, a, 0.*r, v, 1., vc, already_intv, logp)
            else:
                raise NotImplementedError

            # store whether agent has been intervened in current episode
            already_intv |= intervened

            # save and log
            logger.store(VVals=v, VcVals=vc)

            # Update obs (critical!)
            o = next_o

            if intervened:
                if env.intervener.mode == env.intervener.MODE.SAFE_ACTION:
                    while not (d or ep_len == max_ep_len):
                        _, _, _, info = env.step()
                        _, r_safe, d, info_safe = info['safe_step_output']
                        c_safe = info_safe.get('cost', 0.)
                        ep_ret += r_safe
                        ep_cost += c_safe
                        ep_len += 1
                        cum_cost += c_safe
                elif env.intervener.mode == env.intervener.MODE.TERMINATE:
                    pass
                else:
                    raise NotImplementedError

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = (t == local_steps_per_epoch-1)

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len,
                          flush=True)

                # if trajectory didn't reach terminal state, bootstrap value target
                if intervened and env.intervener.mode in [Intervener.MODE.SAFE_ACTION,
                                                          Intervener.MODE.TERMINATE]:
                    v, vc = 0., vc_range[1]
                elif violation:
                    v = 0.
                    vc = (1-ignore_unsafe_cost)*vc_range[1]
                elif timeout or epoch_ended:
                    _, v, vc, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = vc = 0
                buf.finish_path(v, vc)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len,
                                 EpSurrCost=ep_surr_cost)
                o, ep_ret, ep_cost, ep_len = env.reset(), 0, 0, 0
                ep_surr_cost, already_intv = 0, False
                local_episodes += 1
            elif intervened and env.intervener.mode == Intervener.MODE.SAFE_ACTION:
                buf.finish_path(0., vc_range[1])


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env.env}, None)

        # Perform PPO update!
        update(update_penalty_every > 0 and (epoch+1) % update_penalty_every == 0)

        # Cumulative cost calculations
        cumulative_cost = mpi_sum(cum_cost)
        cumulative_surr_cost = mpi_sum(cum_surr_cost)
        cumulative_violations = mpi_sum(cum_viol)
        episodes = mpi_sum(local_episodes)

        cost_rate = cumulative_cost / episodes
        surr_cost_rate = cumulative_surr_cost / episodes
        viol_rate = cumulative_violations / episodes

        # Test the performance of the deterministic version of the agent.
        test_agent()
        o = env.reset()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)

        # Performance
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpCost', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)

        # Cost
        logger.log_tabular('CumulativeCost', cumulative_cost)
        logger.log_tabular('LogCumulativeCost', np.log10(cumulative_cost))
        logger.log_tabular('CostRate', cost_rate)
        logger.log_tabular('EpSurrCost', with_min_and_max=True)
        logger.log_tabular('CumulativeSurrCost', cumulative_surr_cost)
        logger.log_tabular('LogCumulativeSurrCost', np.log10(cumulative_surr_cost))
        logger.log_tabular('SurrCostRate', surr_cost_rate)
        logger.log_tabular('CumulativeViolations', cumulative_violations)
        logger.log_tabular('LogCumulativeViolations', np.log10(cumulative_violations))
        logger.log_tabular('ViolationRate', viol_rate)

        # Test performance
        logger.log_tabular('TestEpRet', with_min_and_max=True)
        logger.log_tabular('TestEpCost', with_min_and_max=True)
        logger.log_tabular('TestEpLen', average_only=True)

        # Penalty
        logger.log_tabular('Penalty', float(penalty_torch.item()))
        logger.log_tabular('LogPenalty', np.log10(float(penalty_torch.item())))

        # Value function stats
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('VcVals', with_min_and_max=True)
        logger.log_tabular('QcVals', with_min_and_max=True)

        # Policy loss and change
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)

        # Value loss and change
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)

        # Policy stats
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)

        # PPO stats
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)

        # Time and steps elapsed
        logger.log_tabular('Time', time.time()-start_time)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.dump_tabular()
