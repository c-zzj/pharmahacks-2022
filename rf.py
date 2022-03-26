import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN


def process_pca(n_components):
    dataset = pd.read_csv("./dataset/challenge_1_gut_microbiome_data.csv")

    X = dataset.iloc[:, 1:-1].to_numpy()
    y = dataset.iloc[:, -1].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)
    ros = RandomOverSampler(random_state=24601)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def rf(X_train, X_val, X_test, y_train, y_val, y_test):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    Xs = [X_train, X_val, X_test]
    ys = [y_train, y_val, y_test]
    y_preds = [clf.predict(X) for X in Xs]
    cf_matrix = [confusion_matrix(y_preds[i], ys[i]) for i in range(3)]
    prefix = ['train', 'val', 'test']

    for i in range(3):
        print(classification_report(ys[i], y_preds[i]))
        plt.figure(i)
        s = np.sum(cf_matrix[i], axis=1)
        cm = cf_matrix[i] / s[:, None]
        sns.heatmap(cm, annot=True, cmap="YlGnBu")
        plt.savefig(f'trained-models/rf/{prefix[i]}_cm.png')
        


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = process_pca(50)
    rf(X_train, X_val, X_test, y_train, y_val, y_test)

