import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset


class DeepFM(nn.Module):
    def __init__(self, config, params, user_profile, item_profile):
        super(DeepFM, self).__init__()
        params = params
        self.device = config['device']

        self.input_dim = config['user_dim'] + config['item_dim']
        self.dnn_hidden_dim1 = params['dnn_hidden_dim1']
        self.dnn_hidden_dim2 = params['dnn_hidden_dim2']
        self.batch_size = params['batch_size']
        self.epoch = params['epoch']
        self.lr = params['lr']

        # DNN part parameters
        self.dnn_hidden_layer1 = nn.Linear(self.input_dim, self.dnn_hidden_dim1)
        self.dnn_hidden_layer2 = nn.Linear(self.dnn_hidden_dim1, self.dnn_hidden_dim2)
        self.dnn_output_layer = nn.Linear(self.dnn_hidden_dim2, 1)

        # FM part parameters
        # - First order part parameters
        self.fm_linear = nn.Linear(self.input_dim, 1)
        # - Second order part parameters
        self.fm_v = nn.Parameter(torch.randn(size=(self.input_dim,)))

        # Conbination part parameters
        self.combine_layer = nn.Linear(2, 2)

        self.user_profile = user_profile
        self.item_profile = item_profile

        self.optimizor = opt.SGD(self.parameters(), lr=self.lr)

    def dnn_part(self, x):
        x = torch.tanh(self.dnn_hidden_layer1(x))
        x = torch.tanh(self.dnn_hidden_layer2(x))
        x = self.dnn_output_layer(x)
        return x

    def fm_part(self, x):
        first_order = self.fm_linear(x)

        tmp1 = torch.matmul(x, self.fm_v.view(-1, 1))
        second_order_term1 = tmp1 * tmp1
        second_order_term2 = torch.matmul(x * x, (self.fm_v * self.fm_v).view(-1, 1))

        return first_order + 0.5 * (second_order_term1 - second_order_term2)

    def forward(self, x):
        dnn_value = self.dnn_part(x)
        fm_value = torch.tanh(self.fm_part(x))

        return self.combine_layer(torch.cat((dnn_value, fm_value), dim=-1))

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
