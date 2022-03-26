import pandas as pd


def split_dataset():
    dataset = pd.read_csv("./dataset/challenge_1_gut_microbiome_data.csv")
    category = dataset.disease.unique()
    DataFrameDict = {elem: pd.DataFrame for elem in category}
    for key in DataFrameDict.keys():
        DataFrameDict[key] = dataset[:][dataset.disease == key]
        df = DataFrameDict[key]
        print(df.shape)
        df = df.loc[:, (df != 0).any(axis=0)]
        print(df.shape)
        df.to_csv(f'datasets/{key}.csv')



if __name__ == "__main__":
    split_dataset()