import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset


class DNN(nn.Module):
    def __init__(self, config, params, user_profile, item_profile):
        super(DNN, self).__init__()
        params = params
        self.device = config['device']

        self.input_dim = config['user_dim'] + config['item_dim']
        self.hidden_dim1 = params['hidden_dim1']
        self.hidden_dim2 = params['hidden_dim2']
        self.batch_size = params['batch_size']
        self.epoch = params['epoch']
        self.lr = params['lr']

        self.hidden_layer1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.hidden_layer2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.output_layer = nn.Linear(self.hidden_dim2, 2)

        self.user_profile = user_profile
        self.item_profile = item_profile

        self.optimizor = opt.SGD(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = torch.tanh(self.hidden_layer1(x))
        x = torch.tanh(self.hidden_layer2(x))
        return self.output_layer(x)

    def train(self, training_domains, valiation_domain):
        training_set = set()
        for d in training_domains:
            training_set |= d
        x_list, y_list = [], []
        for uid, iid, label in training_set:
            x = self.user_profile[uid] + self.item_profile[iid]
            y = label
            x_list.append(x)
            y_list.append(y)
        x_list = torch.tensor(x_list).to(self.device)
        y_list = torch.tensor(y_list).to(self.device)
        training_dataloader = DataLoader(
            TensorDataset(x_list, y_list),
            batch_size=self.batch_size,
            shuffle=True
        )

        x_list, y_list = [], []
        for uid, iid, label in valiation_domain:
            x = self.user_profile[uid] + self.item_profile[iid]
            y = label
            x_list.append(x)
            y_list.append(y)
        x_list = torch.tensor(x_list).to(self.device)
        y_list = torch.tensor(y_list).to(self.device)
        validation_dataloader = DataLoader(
            TensorDataset(x_list, y_list),
            batch_size=self.batch_size,
            shuffle=False
        )

        best_param = self.state_dict()
        min_valid_loss = torch.tensor(100000.0)

        fail_ind = 0
        for e in range(self.epoch):
            exp_loss = torch.tensor(0.0)
            for step, (x, y) in enumerate(training_dataloader):
                score = self.forward(x)
                loss = nn.functional.cross_entropy(score, y)

                self.optimizor.zero_grad()
                loss.backward()
                self.optimizor.step()
                exp_loss += loss.detach().cpu()
            exp_loss /= (step + 1)

            valid_loss = torch.tensor(0.0)
            for step, (x, y) in enumerate(validation_dataloader):
                score = self.forward(x)
                loss = nn.functional.cross_entropy(score, y)
                valid_loss += loss.detach().cpu()
            valid_loss /= (step + 1)

            if valid_loss < min_valid_loss:
                fail_ind = 0
                min_valid_loss = valid_loss
                best_param = self.state_dict()
            else:
                fail_ind += 1

            print('Experience loss(epoch %d): %f, valid loss: %f, min valid loss: %f.' % (
            e, exp_loss, valid_loss, min_valid_loss))

            if fail_ind >= 10:
                print('Early Stop with 10 steps.')
                self.load_state_dict(best_param)
                return min_valid_loss.numpy()
        self.load_state_dict(best_param)
        return min_valid_loss.numpy()

    def predict(self, user_profile, item_profile):
        x = torch.cat([user_profile, item_profile], dim=1).to(self.device)
        return torch.softmax(self.forward(x), dim=1)[:, 1].cpu()
