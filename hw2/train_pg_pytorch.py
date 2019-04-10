import inspect
import os, sys
import time
from itertools import count

import gym
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import Process

import logz


class BaselineData(Dataset):
    def __init__(self, data, label):
        super(BaselineData, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.label[item]


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, n_layers, size,
                 activation=torch.tanh, out_activation=None):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.fc1 = nn.Linear(input_dim, size)
        self.hidden_layers = []
        for i in range(1, n_layers-1):
            self.hidden_layers.append(nn.Linear(size, size))
        self.fc_final = nn.Linear(size, output_dim)
        self.activation = activation
        self.output_activation = out_activation

    def forward(self, x):
        x = self.activation(self.fc1(x))
        for fc in self.hidden_layers:
            x = self.activation(fc(x))
        if self.output_activation is None:
            x = self.fc_final(x)
        else:
            x = self.output_activation(self.fc_final(x))
        return x

    def predict(self, x):
        self.eval()
        x = self.forward(x)
        if x.device != "cpu":
            return x.cpu().data.numpy()
        else:
            return x.data.numpy()


class ContinuePolicy(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, size,
                 activation=torch.tanh, out_activation=None):
        super(ContinuePolicy, self).__init__()
        self.mean = MLP(input_dim, output_dim, n_layers, size,
                        activation, out_activation)
        self.std = nn.Parameter(torch.ones(1, output_dim))

    def forward(self, obs):
        predict_mean = self.mean(obs)
        dist = torch.distributions.Normal(predict_mean, self.std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        return action, log_prob, torch.zeros((log_prob.size(0), 1))


class DiscretePolicy(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, size,
                 activation=torch.tanh, out_activation=None):
        super(DiscretePolicy, self).__init__()
        self.policy = MLP(input_dim, output_dim, n_layers, size,
                          activation, out_activation)

    def forward(self, obs):
        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, dist.entropy().unsqueeze(-1)


def setup_logger(logdir, locals_):
    logz.configure_output_dir(logdir)
    args = inspect.getfullargspec(train_pg)[0][:-1]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)


def quick_log(agent, rewards, iteration, start_time, batch_time_steps,
              total_time_steps):
    returns = [re.sum() for re in rewards]
    ep_length = [len(re) for re in rewards]
    logz.log_dict({"Time": time.time() - start_time,
                   "Iteration": iteration,
                   "AverageReturn": np.mean(returns),
                   "StdReturn": np.std(returns),
                   "MaxReturn": np.max(returns),
                   "MinReturn": np.min(returns),
                   "EpLenMean": np.mean(ep_length),
                   "EpLenStd": np.std(ep_length),
                   "StepsThisBatch": batch_time_steps,
                   "StepsSoFar": total_time_steps
                   })
    logz.dump_tabular()
    logz.save_agent(agent)


def make_agent_args(args, env):
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    args_list = []
    if args["param_search"]:

        lr_search = [0.005, 0.01, 0.02]
        bs_search = [10000, 30000, 50000]
        for lr in lr_search:
            for batch_size in bs_search:
                args_dict = {'n_layers': args['n_layers'],
                             'obs_dim': env.observation_space.shape[0],
                             'act_dim': env.action_space.n if discrete else env.action_space.shape[0],
                             'discrete': discrete,
                             'size': args['size'],
                             'learning_rate': lr,
                             'device': "cuda",
                             'animate': args['render'],
                             'max_path_length': args['ep_len'] if args['ep_len'] > 0 else env.spec.max_episode_steps,
                             'min_batch_time_steps': batch_size,
                             'gamma': args['discount'],
                             'reward_to_go': args['reward_to_go'],
                             'nn_baseline': args['nn_baseline'],
                             'normalize_adv': not args['dont_normalize_advantages'],
                             "n_iter": args["n_iter"],
                             "exp_name": "hc_b{}_r{:.5f}".format(batch_size, lr)}
                args_list.append(args_dict)
    else:
        args_dict = {'n_layers': args['n_layers'],
                     'obs_dim': env.observation_space.shape[0],
                     'act_dim': env.action_space.n if discrete else env.action_space.shape[0],
                     'discrete': discrete,
                     'size': args['size'],
                     'learning_rate': args['learning_rate'],
                     'device': "cuda",
                     'animate': args['render'],
                     'max_path_length': args['ep_len'] if args['ep_len'] > 0 else env.spec.max_episode_steps,
                     'min_batch_time_steps': args['batch_size'],
                     'gamma': args['discount'],
                     'reward_to_go': args['reward_to_go'],
                     'nn_baseline': args['nn_baseline'],
                     'normalize_adv': not args['dont_normalize_advantages'],
                     "n_iter": args["n_iter"],
                     "exp_name": args["exp_name"]}
        args_list.append(args_dict)
    return args_list


def normalize(data):  # Normalize Numpy array or PyTorch tensor to a standard normal distribution.
    return (data - data.mean()) / (data.std() + 1e-7)


class Agent(object):
    def __init__(self, args):
        super(Agent, self).__init__()
        self.obs_dim = args["obs_dim"]
        self.act_dim = args["act_dim"]
        self.discrete = args["discrete"]

        self.size = args["size"]
        self.n_layer = args["n_layers"]
        self.lr = args["learning_rate"]
        if torch.cuda.is_available():
            self.device = torch.device(args["device"])
        else:
            self.device = torch.device("cpu")

        self.gamma = args["gamma"]
        self.animate = args["animate"]
        self.max_path_length = args["max_path_length"]
        self.min_batch_time_steps = args["min_batch_time_steps"]
        self.normalize_adv = args["normalize_adv"]
        self.reward_to_go = args["reward_to_go"]
        self.nn_baseline = args["nn_baseline"]

        if self.discrete:
            self.policy = DiscretePolicy(self.obs_dim, self.act_dim,
                                         self.n_layer, self.size).to(self.device)
        else:
            self.policy = ContinuePolicy(self.obs_dim, self.act_dim,
                                         self.n_layer, self.size).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.policy.eval()

        self.baseline = MLP(self.obs_dim, 1, self.n_layer, self.size).to(self.device)
        self.baseline_criteria = nn.MSELoss()
        self.baseline_optimizer = optim.Adam(self.baseline.parameters(), lr=self.lr)
        self.baseline.eval()

    def sample_trajectory(self, env, animate):
        observations, rewards, actions, log_probs = [], [], [], []
        self.policy.eval()
        obs = env.reset()
        for step in count():
            if animate:
                env.render()
                time.sleep(0.01)
            obs = np.array(obs, dtype=np.float32)
            observations.append(obs)
            act, log_prob, _ = self.policy(torch.from_numpy(obs).to(self.device).unsqueeze(0))
            if self.discrete:
                act = act[0].item()
            else:
                act = act[0].cpu().data.numpy()
            obs, reward, done, _ = env.step(act)
            rewards.append(reward)
            actions.append(act)
            log_probs.append(log_prob.squeeze(-1))
            if done or step > self.max_path_length:
                break
        path = {"observation": np.array(observations, dtype=np.float32),
                "reward": np.array(rewards, dtype=np.float32),
                "action": np.array(actions, dtype=np.float32),
                "log_prob": torch.cat(log_probs)}
        return path

    def sample_trajectories(self, iteration, env):
        batch_time_steps = 0
        paths = []
        while batch_time_steps <= self.min_batch_time_steps:
            animate_this_episode = (len(paths) == 0 and (iteration % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            batch_time_steps += len(path["reward"])
        return paths

    def sum_of_reward(self, rewards):
        """
            Monte Carlo estimation of the Q function.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                re_n: length: num_paths. Each element in re_n is a numpy array
                    containing the rewards for the particular path

            returns:
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths

            ----------------------------------------------------------------------------------

            Your code should construct numpy arrays for Q-values which will be used to compute
            advantages (which will in turn be fed to the placeholder you defined in
            Agent.define_placeholders).

            Recall that the expression for the policy gradient PG is

                  PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]

            where

                  tau=(s_0, a_0, ...) is a trajectory,
                  Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
                  and b_t is a baseline which may depend on s_t.

            You will write code for two cases, controlled by the flag 'reward_to_go':

              Case 1: trajectory-based PG

                  (reward_to_go = False)

                  Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over
                  entire trajectory (regardless of which time step the Q-value should be for).

                  For this case, the policy gradient estimator is

                      E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]

                  where

                      Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.

                  Thus, you should compute

                      Q_t = Ret(tau)

              Case 2: reward-to-go PG

                  (reward_to_go = True)

                  Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
                  from time step t. Thus, you should compute

                      Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}


            Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
            like the 'ob_no' and 'ac_na' above.
        """
        q_n = []
        if self.reward_to_go:
            for reward in rewards:
                q_path = reward.copy()
                for i in range(1, reward.shape[0]):
                    reward = reward[1:] * self.gamma
                    q_path[:-i] += reward
                q_n.append(q_path)
        else:
            for reward in rewards:
                gammas = np.ones_like(reward) * self.gamma
                gammas[0] = 1
                gammas = np.cumprod(gammas)
                q_path = np.sum(reward * gammas) * np.ones_like(reward)
                q_n.append(q_path)
        return np.concatenate(q_n)

    def compute_advantage(self, ob_no, q_n):
        q_n = torch.from_numpy(q_n).to(self.device)
        if self.nn_baseline:
            obs_no = torch.from_numpy(ob_no).to(self.device)
            self.baseline.eval()
            b_n = self.baseline(obs_no).squeeze()
            b_n = normalize(b_n) * q_n.std() + q_n.mean()
            adv_n = q_n - b_n.type_as(q_n)
        else:
            adv_n = q_n
        return adv_n

    def estimate_return(self, ob_no, re_n):
        """
            Estimates the returns over a set of trajectories.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                re_n: length: num_paths. Each element in re_n is a numpy array
                    containing the rewards for the particular path

            returns:
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
        """
        q_n = self.sum_of_reward(re_n)
        adv_n = self.compute_advantage(ob_no, q_n)
        if self.normalize_adv:
            adv_n = normalize(adv_n)
        return q_n, adv_n

    def update_parameters(self, obs_no, ac_na, q_n, adv_n, log_probs):
        if self.nn_baseline:
            obs_no = torch.from_numpy(obs_no).to(self.device)
            target = torch.from_numpy(q_n).to(self.device).unsqueeze(-1)
            self.baseline.train()
            self.baseline_optimizer.zero_grad()
            prediction = self.baseline(obs_no)
            baseline_loss = self.baseline_criteria(input=prediction, target=target)
            baseline_loss.backward()
            self.baseline_optimizer.step()

        self.policy.train()
        self.policy_optimizer.zero_grad()
        loss = -torch.mean(log_probs * adv_n)
        loss.backward()
        self.policy_optimizer.step()


def train_pg(args, log_dir, seed, env):
    start_time = time.time()
    setup_logger(log_dir, locals())

    # Setup random seed
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = Agent(args)

    total_time_steps = 0
    for iteration in range(args["n_iter"]):
        print("*"*10+" Iteration {} ".format(iteration)+"*"*10)
        paths = agent.sample_trajectories(iteration, env)
        obs, rewards, action, log_probs = [], [], [], []
        for path in paths:
            obs.append(path["observation"])
            rewards.append(path["reward"])
            action.append(path["action"])
            log_probs.append(path["log_prob"])
        obs = np.concatenate(obs)
        action = np.concatenate(action)
        log_probs = torch.cat(log_probs)
        q_n, adv_n = agent.estimate_return(obs, rewards)
        agent.update_parameters(obs, action, q_n, adv_n, log_probs)
        batch_time_steps = obs.shape[0]
        total_time_steps += batch_time_steps
        quick_log(agent=agent, rewards=rewards, iteration=iteration, start_time=start_time,
                  batch_time_steps=batch_time_steps, total_time_steps=total_time_steps)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument("--param_search", "-ps", action="store_true")
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    # Setup gym
    env = gym.make(args.env_name)

    agent_args = make_agent_args(vars(args), env)
    for agent_arg in agent_args:
        if len(agent_args) == 1:
            log_dir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        else:
            log_dir = 'hc_b{}_r{:.5f}_{}'.format(agent_arg["min_batch_time_steps"],
                                                 agent_arg["learning_rate"],
                                                 time.strftime("%d-%m-%Y_%H-%M-%S"))

        log_dir = os.path.join(data_path, log_dir)
        if not (os.path.exists(log_dir)):
            os.makedirs(log_dir)

        processes = []
        for e in range(args.n_experiments):
            seed = args.seed + 10 * e
            print('Running experiment with seed %d' % seed)
            p = Process(target=train_pg, args=(agent_arg,
                                               os.path.join(log_dir, '%d' % seed),
                                               seed, env))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == '__main__':
    main()
