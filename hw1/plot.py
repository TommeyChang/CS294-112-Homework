import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

import argparse


def vis(pkl_data):
    fp = open(pkl_data, "rb")
    results = pkl.load(fp)

    data_mean = [[] for _ in range(11)]
    data_std = [[] for _ in range(11)]
    label = [[] for _ in range(11)]
    for item in results:
        data_idx = item["dagger time"]
        training_iter = item["training epoch"]
        label[data_idx].append(training_iter)
        mean = np.mean(item["returns"])
        std = np.std(item["returns"])
        data_mean[data_idx].append(mean)
        data_std[data_idx].append(std)

    ind = np.arange(len(data_std[0]))
    width = 0.08
    colors = ["peru", "dodgerblue", "teal", "brown", "darkslategrey",
              "lightsalmon", "fuchsia", "greenyellow", "skyblue", "darkorange",
              "seagreen"]

    fig, ax = plt.subplots()
    idx = 0
    for mean, std in zip(data_mean, data_std):
        ax.bar(ind - (5 - idx) * width, mean, width, yerr=std,
               color=colors[idx], label="DAgger {}".format(idx))
        idx += 1
    ax.set_ylabel('Reward')
    ax.set_title('Mean and std of rewards with different settings')
    ax.set_xticks(ind)
    ax.set_xticklabels(label[0])
    ax.set_xlabel("Training iter")
    ax.legend()

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file_name", type=str,
                        help="Log file name")
    args = parser.parse_args()
    vis(args.log_file_name)


if __name__ == '__main__':
    main()
