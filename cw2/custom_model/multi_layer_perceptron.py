import torch as t
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from .anlp_model import AnlpModel
from sklearn.utils import compute_class_weight
import logging

INPUT_SIZE = 768
NUM_CLASSES = 17
LAYER_SIZES = [500, 500, NUM_CLASSES]
BATCH_SIZE = 32
NUM_EPOCHS = 100
STEP_SIZE = 30
LR = 1e-2
MOMENTUM = 0


class MultiLayerPerceptron(AnlpModel):
    def __init__(self,
                 input_size=INPUT_SIZE,
                 layer_sizes=LAYER_SIZES):
        modules = []
        prev_size = input_size
        for i, layer in enumerate(layer_sizes):
            modules.append(t.nn.Linear(prev_size, layer))
            modules.append(t.nn.ReLU())
            if i + 1 != len(layer_sizes):
                modules.append(t.nn.Dropout())
            prev_size = layer
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.actual_model = t.nn.Sequential(*modules).to(self.device)

    def fit(self, train_data, train_labels, eval_data, eval_labels):
        weights = compute_class_weight('balanced',
                                       classes=np.unique(train_labels),
                                       y=train_labels)
        optim = t.optim.SGD(self.actual_model.parameters(),
                            lr=LR,
                            momentum=MOMENTUM)
        scheduler = t.optim.lr_scheduler.StepLR(optim, STEP_SIZE)
        train_data = t.Tensor(train_data)
        train_labels = t.Tensor(train_labels).to(t.int64)
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        eval_data = t.Tensor(eval_data)
        eval_labels = t.Tensor(eval_labels).to(t.int64)
        eval_dataset = TensorDataset(eval_data, eval_labels)
        eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
        criterion = t.nn.CrossEntropyLoss(weight=t.Tensor(weights)) \
            .to(self.device)
        for e in range(NUM_EPOCHS):
            print(f"Epoch {e + 1}:")
            for train in [True, False]:
                if train:
                    self.actual_model.train()
                else:
                    self.actual_model.eval()
                total_corrects = 0
                total_loss = 0
                data = train_data if train else eval_data
                data_loader = train_loader if train else eval_loader
                for batch, actual in tqdm(data_loader):
                    batch, actual = \
                        batch.to(self.device), actual.to(self.device)
                    t.set_grad_enabled(train)
                    optim.zero_grad()
                    predicted = self.actual_model(batch)
                    loss = criterion(predicted, actual)
                    if train:
                        loss.backward()
                        optim.step()
                    total_loss += loss.item() * len(batch)
                    predicted_softmax = t.argmax(predicted, dim=1)
                    total_corrects += (actual == predicted_softmax).sum()
                logging.info("Train" if train else "Eval")
                logging.info(f"Accuracy: {total_corrects/len(data)}")
                logging.info(f"Total loss: {total_loss}")
            scheduler.step()

    def predict(self, data):
        self.actual_model.eval()
        with t.no_grad():
            data = t.Tensor(data)
            dataset = TensorDataset(data)
            data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
            preds = []
            for batch in data_loader:
                batch = batch[0].to(self.device)
                predicted = self.actual_model(batch)
                preds.append(t.argmax(predicted, dim=1).cpu().numpy())
            return np.concatenate(preds)
