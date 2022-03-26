from typing import Callable, Dict

from torch import device
from torch.nn import Module, CrossEntropyLoss
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor

from data import *

TrainingPlugin = Callable[[Any, int], None]
Metric = Callable[[Any, DataLoader], float]  # pred, true -> result. The higher the better


class Function:
    @staticmethod
    def flatten(x: Tensor):
        """
        flatten the tensor
        :param x:
        :return:
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return x.view(-1, num_features)


class Classifier:
    """
    Abstract Classifier
    """

    def __init__(self,
                 training_l: LabeledDataset,
                 validation: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None,
                 ):
        self.device = device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.training_l = training_l
        self.validation = validation
        self.training_ul = training_ul

    def train_performance(self, metric: Metric, proportion: float = 0.025, batch_size=300):
        """
        Obtain the performance on a subset of the training set
        :param metric: the metric of performance
        :param proportion: the proportion of the subset to be checked
        :param batch_size: size of the batch of each prediction (for solving the GPU out-of-memory problem)
        :return:
        """
        proportion_to_check = int(proportion * len(self.training_l))
        t, _ = random_split(self.training_l, [proportion_to_check, len(self.training_l) - proportion_to_check])
        loader = DataLoader(t, batch_size=batch_size, shuffle=False)
        return metric(self, loader)

    def val_performance(self, metric: Metric, batch_size=300):
        """
        Obtain the performance on a subset of the training set
        :param metric: the metric of performance
        :param batch_size: size of the batch of each prediction (for solving the GPU out-of-memory problem)
        :return:
        """
        loader = DataLoader(self.validation, batch_size=batch_size, shuffle=False)
        return metric(self, loader)

    def evaluate(self, dataset: LabeledDataset, metric: Metric, batch_size=300):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return metric(self, loader)

    def predict(self, x: Tensor):
        """
        to be implemented by concrete classifiers
        :param x:
        :return:
        """
        raise NotImplementedError


class OptimizerProfile:
    def __init__(self, optimizer: Callable[..., Optimizer],
                 parameters: Dict[str, Any] = {}):
        self.optim = optimizer
        self.params = parameters


class NNClassifier(Classifier):
    """
    Abstract Network Classifier
    """

    def __init__(self,
                 network: Callable[..., Module],
                 training_l: LabeledDataset,
                 validation: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None,
                 network_params: Dict[str, Any] = {},
                 ):
        """
        :param network: a function that gives a Network
        :param training_l: the labeled data
        :param validation: the validation set
        :param training_ul: (optional) the unlabeled data
        """
        super(NNClassifier, self).__init__(training_l, validation, training_ul)
        self.network = network(**network_params).to(self.device)
        self.optim = SGD(self.network.parameters(), lr=1e-3, momentum=0.99)
        self.loss = CrossEntropyLoss()
        self.training_message = 'No training message.'
        # temporary variable used for plugins to communicate
        self._tmp = {}

    def load_network(self,
                     folder_path: Path,
                     epoch: int):
        """
        :param network:
        :param folder_path:
        :param epoch:
        :return: a network callable that can be passed to the NNClassifier constructor
        """
        self.network.load_state_dict(torch.load(folder_path / f"{epoch}.params"))
        if (folder_path / "performance.pt").exists():
            self._tmp['learning_path'] = torch.load(folder_path / "performance.pt")

    def set_optimizer(self, optimizer: OptimizerProfile):
        self.optim = optimizer.optim(self.network.parameters(), **optimizer.params)

    def set_loss(self, loss: Callable):
        self.loss = loss

    def train(self,
              epochs: int,
              batch_size: int,
              shuffle: bool = True,
              start_epoch: int = 1,
              plugins: Optional[List[TrainingPlugin]] = None,
              verbose: bool = True) \
            -> None:
        """
        Train the model up to the epochs given.
        There is no return value. Plugins are used to save model and record performances.
        :param verbose:
        :param start_epoch:
        :param epochs: number of epochs
        :param batch_size: batch size for training
        :param shuffle: whether or not to shuffle the training data
        :param plugins: training plugin that is run after each epoch
        :return: None
        """
        if verbose:
            s = ''
            s += "Model Summary:\n"
            s += repr(self.network) + '\n'
            s += f"Device used for training: {self.device}\n"
            s += f"Size of training set: {len(self.training_l)}\n"
            s += f"Size of validation set: {len(self.validation)}\n"
            self.training_message = s
            print(s)
        train_loader = DataLoader(self.training_l, batch_size=batch_size, shuffle=shuffle)
        # the following code adopted from the tutorial notebook
        for epoch in range(start_epoch, start_epoch + epochs):  # loop over the data multiple times
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optim.zero_grad()

                # forward + backward + optimize
                self.loss.zero_grad()
                outputs = self.network(inputs.float())
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optim.step()
            if verbose:
                s = f"---{epoch} EPOCHS FINISHED---\n"
                self.training_message += s
                print(s, end='')
            if plugins:
                s = f"Plugin messages for epoch {epoch}:\n"
                self.training_message += s
                print(s, end='')
                for plugin in plugins:
                    plugin(self, epoch)
                self.training_message = ''  # reset training message
        if verbose:
            s = f"\nFinished training all {epochs} epochs."
            self.training_message = s
            print(s)
        return

    def predict(self, x: Tensor):
        """
        to be implemented by concrete networks
        :param x:
        :return:
        """
        return self._pred(x)

    def _pred(self, x: Tensor):
        """
        argmax prediction - use the highest value as prediction
        :param x:
        :return:
        """
        self.network.eval()
        with torch.no_grad():
            pred = self.network(x.float())
        self.network.train()

        indices = torch.argmax(pred, dim=1)
        return indices
