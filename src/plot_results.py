import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join
import argparse
import numpy as np


def extract_data(files):
    res = []
    for file in files:
        df = pd.read_csv(file).iloc[:,0]
        df = df.apply(lambda x: x[1: -1]).to_numpy().astype(float) 
        res.append(df)
    return res

def calculate_win_rate(data):
    counter = 0
    num_of_trials = len(data)
    win_rate = []

    for i in range(1, num_of_trials + 1):
        if data[i - 1] > 0:
            counter += 1
        win_rate.append(counter / i * 100)
    return win_rate

def plot_results(data):
    win_rates = [calculate_win_rate(d) for d in data]
    x_data = np.arange(1, len(win_rates[0]) + 1)

    for win_rate in win_rates:
        plt.plot(x_data, win_rate)

    plt.legend(["a2c_easy_difficulty", "a2c_hard_difficulty", "a2c_normal_difficulty"])
    plt.xlabel("trials")
    plt.ylabel("win rate %")
    plt.show()
        

# if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()
onlyfiles = [join(args.path, f) for f in listdir(args.path) if isfile(join(args.path, f))]
data = extract_data(onlyfiles)
win_rates = [calculate_win_rate(d) for d in data]
x_data = np.arange(1, len(win_rates[0]) + 1)
for win_rate in win_rates:
    plt.plot(x_data, win_rate)
plt.legend(["a2c_easy_difficulty", "a2c_hard_difficulty", "a2c_normal_difficulty"])
plt.xlabel("trials")
plt.ylabel("win rate %")
plt.show()


