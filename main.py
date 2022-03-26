from classifier.network.simple_mlp import *
from classifier.plugin import *
from classifier.metric import *
from torch.optim import Adam
from typing import Callable, Dict
import pandas
import matplotlib.pyplot as plt
from sklearn import preprocessing

import numpy as np

TRAINED_MODELS_PATH = Path("trained-models")



def data_init(path: str = './dataset',
              dataset_fname: str = 'challenge_1_gut_microbiome_data.csv',
              rand_seed: int = 0,
              train_val_test_split: List[int] = [7, 1, 2]):
    torch.manual_seed(rand_seed)
    dataset = pandas.read_csv(Path(path) / dataset_fname)
    le = preprocessing.LabelEncoder()
    le.fit(dataset['disease'])
    dataset['disease'] = le.transform(dataset['disease'])

    x = dataset.iloc[:, 1:-1].to_numpy()
    y = dataset.iloc[:, -1].to_numpy()

    x = torch.from_numpy(x)
    y = torch.from_numpy(y).type(torch.LongTensor)

    dataset = LabeledDataset(x, y)

    train_size = int((train_val_test_split[0] / sum(train_val_test_split)) * len(dataset))
    val_size = int((train_val_test_split[1] / sum(train_val_test_split)) * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train, val, test = random_split(dataset, [train_size, val_size, test_size])
    #
    torch.save(train, Path(path) / 'train')
    torch.save(val, Path(path) / 'val')
    torch.save(test, Path(path) / 'test')


def load_data(path: str = './dataset'):
    train = torch.load(Path(path) / 'train')
    val = torch.load(Path(path) / 'val')
    test = torch.load(Path(path) / 'test')
    return train, val, test


ADAM_PROFILE = OptimizerProfile(Adam, {
    "lr": 0.0005,
    "betas": (0.9, 0.99),
    "eps": 1e-8
})

SGD_PROFILE = OptimizerProfile(SGD, {
    'lr': 0.0005,
    'momentum': 0.99
})


def train_model(model: Callable[..., Module], fname: str, model_params: Dict[str, Any] = {},
                epochs: int = 100,
                continue_from: int = 0,
                batch_size: int = 30):
    """

    :param model:
    :param fname: name of the model
    :param model_params: parameters to be passed to the model
    :param epochs: number of epochs to run
    :param continue_from: number of epoch to start from
    :param batch_size:
    :return:
    """
    print(fname)
    print(model)
    print(model_params)

    model_path = Path(TRAINED_MODELS_PATH / fname)

    clf = NNClassifier(model, TRAIN, VAL, network_params=model_params)

    conv_params = sum(p.numel() for p in clf.network.layers.parameters() if p.requires_grad)
    print(conv_params)

    print(f"Epochs to train: {epochs}")
    print(f"Continue from epoch: {continue_from}")
    if continue_from > 0:
        clf.load_network(model_path, continue_from)

    clf.set_optimizer(ADAM_PROFILE)

    clf.train(epochs,
              batch_size=batch_size,
              plugins=[
                  save_good_models(model_path),
                  calc_train_val_performance(accuracy),
                  print_train_val_performance(accuracy),
                  log_train_val_performance(accuracy),
                  save_training_message(model_path),
                  plot_train_val_performance(model_path, 'Modified AlexNet', accuracy, show=False,
                                             save=True),
                  elapsed_time(),
                  save_train_val_performance(model_path, accuracy),
              ],
              start_epoch=continue_from + 1
              )


def get_best_epoch(fname: str):
    """
    get the number of best epoch
    chosen from: simplest model within 0.001 acc of the best model
    :param fname:
    :return:
    """
    model_path = Path(TRAINED_MODELS_PATH / fname)
    performances = load_train_val_performance(model_path)
    epochs = performances['epochs']
    val = performances['val']
    highest = max(val)
    index_to_chose = -1
    for i in range(len(val)):
        if abs(val[i] - highest) < 0.001:
            index_to_chose = i
            print(f"Val acc of model chosen: {val[i]}")
            break
    return epochs[index_to_chose]


def obtain_test_acc(model: Callable[..., Module], fname: str, model_params: Dict[str, Any] = {}, *args, **kwargs):
    best_epoch = get_best_epoch(fname)
    clf = NNClassifier(model, None, None, network_params=model_params)
    model_path = Path(TRAINED_MODELS_PATH / fname)
    clf.load_network(model_path, best_epoch)
    acc = clf.evaluate(TEST, accuracy)
    print(f"\nTEST SET RESULT FOR {fname}: {acc}\n")


def train_and_test(model: Callable[..., Module], fname: str, model_params: Dict[str, Any] = {},
                   epochs: int = 100,
                   continue_from: int = 0,
                   batch_size: int = 100
                   ):
    train_model(model, fname, model_params, epochs, continue_from, batch_size)
    obtain_test_acc(model, fname, model_params)


def plot_acc(entries: Dict[str, str], title: str, target: str, epochs_to_show: int = 50, plot_train: bool = False,
             show: bool = False):
    """

    :param entries: dict of the form {file_name: label}
    :param title: title of the plot
    :param target: target file name to save the plot
    :param epochs_to_show:
    :param show:
    :return:
    """
    plt.figure()
    for k in entries:
        model_path = Path(TRAINED_MODELS_PATH / k)
        performances = load_train_val_performance(model_path)
        index = -1
        epochs = performances['epochs']
        for i in epochs:
            if epochs[i] == epochs_to_show:
                index = i
                break
        epochs = epochs[:index]
        val = performances['val'][:index]
        train = performances['train'][:index]
        if plot_train:
            plt.plot(epochs, val,
                     label=entries[k] + '-val', alpha=0.5)
            plt.plot(epochs, train,
                     label=entries[k] + '-train', alpha=0.5)
        else:
            plt.plot(epochs, val,
                     label=entries[k], alpha=0.5)
    # plt.ylim(bottom=0.5)
    plt.xlabel('Number of epochs')
    if plot_train:
        plt.ylabel('Accuracy')
    else:
        plt.ylabel('Validation accuracy')
    plt.title(title)
    plt.legend()
    plt.savefig(TRAINED_MODELS_PATH / target)
    if show:
        plt.show()


def experiment(epochs: int = 50):
    to_run = (
        (SimpleMLP, "simple-mlp"),
    )

    for p in to_run:
        print(type(p))
        train_and_test(*p)

    entries = {
        'simple-mlp': 'MLP'
    }
    plot_acc(entries, 'Simple MLP', 'simple-mlp.jpg', epochs, plot_train=True)


if __name__ == '__main__':
    #data_init()
    TRAIN, VAL, TEST = load_data()
    TRAINED_MODELS_PATH = Path("trained-models")
    experiment()


