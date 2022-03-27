import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def split_dataset():
    dataset = pd.read_csv("./dataset/challenge_1_gut_microbiome_data.csv")
    category = dataset.disease.unique()
    DataFrameDict = {elem: pd.DataFrame for elem in category}
    means = []
    vars = []
    for key in DataFrameDict.keys():
        DataFrameDict[key] = dataset[:][dataset.disease == key]
        df = DataFrameDict[key]
        df = df.drop(df.columns[0], axis=1)
        df = df.drop(df.columns[-1], axis=1)
        mean = df.mean(axis=0).to_numpy()
        means.append(mean)
        var = df.var(axis=0).to_numpy()
        vars.append(var)

        # print(df.shape)
        # df = df.loc[:, (df != 0).any(axis=0)]
        # print(df.shape)
        # df.to_csv(f'datasets/{key}.csv')

    fig, axs = plt.subplots(4)

    x = np.arange(1, 1095)
    legends = ["Disease-1", "Disease-2", "Disease-3", "Healthy"]
    # for i in range(4):
    #     axs[i].plot(x, vars[i], label = legends[i])
    #     # axs[i].legend()
    # fig.savefig('var_nl.png')
    for i in range(4):
        axs[i].plot(x, means[i], label = legends[i])
        # axs[i].legend()
    fig.savefig('mean_nl.png')



if __name__ == "__main__":
    split_dataset()
    #df1 = pd.read_csv("./datasets/Disease-1.csv")
