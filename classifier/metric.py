from classifier import *
from sklearn.metrics import f1_score


def accuracy(clf: Classifier, data_loader: DataLoader, ) -> float:
    total = 0
    correct = 0
    for i, data in enumerate(data_loader, 0):
        x = data[0].to(clf.device)
        pred = clf.predict(x)
        true = data[1].to(clf.device)
        total += data_loader.batch_size
        correct += (pred == true).sum().item()
    return correct / total


def get_f1(average: str = 'macro') -> Metric:
    def f1(clf: Classifier, data_loader: DataLoader, ) -> float:
        true = []
        pred = []
        for i, data in enumerate(data_loader, 0):
            x = data[0].to(clf.device)
            pred += clf.predict(x).tolist()
            true += data[1].to(clf.device).tolist()
        return f1_score(true, pred, average=average)
    return f1
