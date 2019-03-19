#!/usr/bin/env python
# coding: utf-8
"""
This is a pytorch implementation of CS294-112 homework 1
Thanks for the code in https://github.com/PengZhenghao/CS294-Homework
"""

import argparse
import pickle as pkl
import matplotlib.pylab as plt

import gym
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset

import tf_util
from load_policy import load_policy

env_names = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2",
             "Humanoid-v2", "Reacher-v2", "Walker2d-v2"]


# model definition

class BC(nn.Module):
    
    def __init__(self, ops_dim, actn_dim):
        super(BC, self).__init__()

        self.fc1 = nn.Linear(ops_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, actn_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# dataset loader
class Rollout(Dataset):
    def __init__(self, env_name):
        assert env_name in env_names
        exp_data_name = 'expert_data/{}.pkl'.format(env_name)
        fp = open(exp_data_name, 'rb')
        data = pkl.load(fp)

        obs = data["observations"].squeeze()
        actions = data["actions"].squeeze()

        self.data = obs.astype(np.float32)
        self.label = actions.astype(np.float32)

    def aggregate(self, obs, actions):
        self.data = np.concatenate([self.data, obs.squeeze()])
        self.label = np.concatenate([self.label, actions.squeeze()])

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)

    def get_dim(self):
        return self.data.shape[-1], self.label.shape[-1]


# train iter
def train_iter(args, model, criterion, device, train_loader,
               optimizer, batch_count, writer, epoch_idx):

    model.train()
    iter_count = 0

    total_loss = .0
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_count += 1
        iter_count += 1

        data_tensor = data.to(device)
        label_tensor = target.to(device)
        optimizer.zero_grad()
        out = model(data_tensor)
        loss = criterion(out, label_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_count % args.log_interval == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), batch_count)
            writer.add_scalar("loss", loss.item(), batch_count)

        if epoch_idx % args.print_interval == 0:
            print("Epoch :{}, average loss is: {}".format(epoch_idx, total_loss / iter_count))

        return batch_count


def run_learning_policy(env, model, device, episode, render):
    model.eval()
    returns = []
    observations = []
    actions = []
    for i in range(episode):
        obs = env.reset()
        done = False
        total_reward = 0.
        steps = 0
        while not done:
            obs_tensor = torch.tensor(obs[None, :], dtype=torch.float).to(device)
            action = model(obs_tensor).cpu().data.numpy()
            obs, r, done, _ = env.step(action)
            total_reward += r
            steps += 1
            observations.append(obs)
            actions.append(action)
            if render:
                env.render()
        returns.append(total_reward)

    return observations, actions, returns, steps


def data_aggr(env, episode, learn_policy, device, expert_policy, dataset, dagger_idx):
    print("The {} time data aggregate.".format(dagger_idx))
    observations, _, _, _ = run_learning_policy(env, learn_policy, device, episode, False)
    actions = expert_policy(observations)
    observations = np.array(observations).astype(np.float32)
    actions = np.array(actions).astype(np.float32)
    dataset.aggregate(observations, actions)
    print("Data aggregate successfully")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str)

    parser.add_argument("--batch_size", type=int, default=128, metavar="N",
                        help="Input batch size for training (default: 128)")
    parser.add_argument("--lr", type=float, default=0.0005, metavar="LR",
                        help="Learning rate (default: 0.0005)")
    parser.add_argument("--epoch", type=int, default=200, metavar="N",
                        help="Number of epoch to train (default: 100)")
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="Disable GPU")
    parser.add_argument("--print_interval", type=int, default=20, metavar="N",
                        help="Training status print interval")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N",
                        help="Log status interval")

    parser.add_argument("--dagger_episodes", type=int, default=50, metavar="N",
                        help="Number of episodes in collecting DAgger data (default: 1)")
    parser.add_argument("--dagger_time", type=int, default=1, metavar="N",
                        help="DAgger times in training (default: 1)")

    parser.add_argument("--test_interval", type=int, default=50,
                        help="Test model interval after training epochs")
    parser.add_argument("--test_episode", type=int, default=10,
                        help="Episode while test")
    args = parser.parse_args()
    env_name = args.env_name

    # set up device

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # setup training data set
    train_data = Rollout(env_name)
    obs_dim, action_dim = train_data.get_dim()
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, **kwargs)

    # set up cloning policy and expert policy
    cloning_net = BC(obs_dim, action_dim).to(device)
    writer = SummaryWriter(log_dir='runs')

    expert_policy = load_policy("experts/" + args.env_name + ".pkl")

    # setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cloning_net.parameters(), lr=args.lr)

    # setup RL environment
    env = gym.make(env_name)

    # tf session for running expert
    with tf.Session():
        tf_util.initialize()
        # DAgger phase
        phase_num = args.dagger_time + 1
        test_result = []

        batch_count = 0
        dagger_idx = 0
        while phase_num > 0:
            phase_num -= 1
            print("Data count: {}".format(len(train_data)))

            for epoch_idx in range(args.epoch):
                batch_count = train_iter(args, cloning_net, criterion, device, train_loader,
                                         optimizer, batch_count, writer, epoch_idx)
                if epoch_idx % args.test_interval == 0:
                    _, _, returns, steps = run_learning_policy(env, cloning_net, device,
                                                        args.test_episode, False)
                    result = {"dagger time": dagger_idx, "training epoch": epoch_idx,
                              "returns": returns, "steps": steps}
                    test_result.append(result)
            if phase_num > 0:
                dagger_idx += 1
                data_aggr(env, args.dagger_episodes, cloning_net, device,
                          expert_policy, train_data, dagger_idx)

    _, _, returns, _ = run_learning_policy(env, cloning_net, device, 20, True)
    mean = np.mean(returns)
    std = np.std(returns)
    print("Test returns: {}, \n mean: {}, std: {}".format(returns, mean, std))

    torch.save(cloning_net, "behavioral_cloning_{}.pth".format(env_name))
    writer.export_scalars_to_json('./log/all_scalar_{}.json'.format(env_name))

    fp = open("{}_valid_log.pkl".format(env_name), "wb")
    pkl.dump(test_result, fp)


if __name__ == '__main__':
    main()
