import random

import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset


class MLP_handle(nn.Module):
    def __init__(self, config, params):
        super(MLP_handle, self).__init__()

        self.input_dim = config['user_dim'] + config['item_dim']
        self.hidden_dim = params['hidden_dim']

        self.hidden_layer_weight = nn.Parameter(torch.randn(size=(self.input_dim, self.hidden_dim)))
        self.hidden_layer_bias = nn.Parameter(torch.randn(size=(self.hidden_dim,)))

        self.output_layer_weight = nn.Parameter(torch.randn(size=(self.hidden_dim, 2)))
        self.output_layer_bias = nn.Parameter(torch.randn(size=(2,)))

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(torch.matmul(x, self.hidden_layer_weight) + self.hidden_layer_bias)
        x = torch.matmul(x, self.output_layer_weight) + self.output_layer_bias
        return x


class MLP_dispatch(nn.Module):
    def __init__(self, config, params):
        super(MLP_dispatch, self).__init__()

        self.shared_mlp = MLP_handle(config, params)
        self.specific_mlp_list = [MLP_handle(config, params).to(config['device']) for i in
                                  range(config['num_source_domain'] - 1)]

        self.shared_optimizor = torch.optim.SGD(self.shared_mlp.parameters(), lr=params['lr'])
        self.specific_optimizor_list = [torch.optim.SGD(mlp.parameters(), lr=params['lr']) for mlp in
                                        self.specific_mlp_list]

    def shared_forward(self, x):
        return self.shared_mlp(x)

    def specific_forward(self, x, did):
        return self.specific_mlp_list[did](x)

    def inference_forward(self, x):
        hidden_layer_weight = self.shared_mlp.hidden_layer_weight + torch.mean(
            torch.stack([mlp.hidden_layer_weight for mlp in self.specific_mlp_list], dim=0), dim=0)
        hidden_layer_bias = self.shared_mlp.hidden_layer_bias + torch.mean(
            torch.stack([mlp.hidden_layer_bias for mlp in self.specific_mlp_list], dim=0), dim=0)

        output_layer_weight = self.shared_mlp.output_layer_weight + torch.mean(
            torch.stack([mlp.output_layer_weight for mlp in self.specific_mlp_list], dim=0), dim=0)
        output_layer_bias = self.shared_mlp.output_layer_bias + torch.mean(
            torch.stack([mlp.output_layer_bias for mlp in self.specific_mlp_list], dim=0), dim=0)

        x = torch.nn.functional.leaky_relu(torch.matmul(x, hidden_layer_weight) + hidden_layer_bias)
        x = torch.matmul(x, output_layer_weight) + output_layer_bias
        return x


class MAMDR(nn.Module):
    def __init__(self, config, params, user_profile, item_profile):
        super(MAMDR, self).__init__()
        self.device = config['device']

        self.params = params
        self.mlp_dispatch = MLP_dispatch(config, params)

        self.batch_size = params['batch_size']
        self.epoch = params['epoch']
        self.lr = params['lr']

        self.user_profile = user_profile
        self.item_profile = item_profile

    def forward(self, x):
        return self.mlp_dispatch.inference_forward(x)

    def train(self, training_domains, valiation_domain):
        training_dataloader_list = []
        for d in training_domains:
            x_list, y_list = [], []
            for uid, iid, label in d:
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
            training_dataloader_list.append(training_dataloader)

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

            # Domain Negotiation(DN)
            cnt = 0
            saved_params = self.mlp_dispatch.shared_mlp.state_dict()
            for dataloader in training_dataloader_list:
                for step, (x, y) in enumerate(dataloader):
                    score = self.mlp_dispatch.shared_forward(x)
                    loss = nn.functional.cross_entropy(score, y)

                    self.mlp_dispatch.shared_optimizor.zero_grad()
                    loss.backward()
                    self.mlp_dispatch.shared_optimizor.step()
                    exp_loss += loss.detach().cpu()
                    cnt += 1
            current_params = self.mlp_dispatch.shared_mlp.state_dict()
            for name, param in saved_params.items():
                saved_params[name] = (1 - self.params['beta']) * saved_params[name] + self.params['beta'] * \
                                     current_params[name]
            self.mlp_dispatch.shared_mlp.load_state_dict(saved_params)
            # Domain Regression(DG)
            for did, dataloader in enumerate(training_dataloader_list):
                extra_domain = did
                while extra_domain == did:
                    extra_domain = random.randint(0, len(training_dataloader_list) - 1)

                saved_params = self.mlp_dispatch.specific_mlp_list[did].state_dict()
                for step, (x, y) in enumerate(dataloader):
                    score = self.mlp_dispatch.specific_forward(x, did)
                    loss = nn.functional.cross_entropy(score, y)

                    self.mlp_dispatch.specific_optimizor_list[did].zero_grad()
                    loss.backward()
                    self.mlp_dispatch.specific_optimizor_list[did].step()
                    exp_loss += loss.detach().cpu()
                    cnt += 1
                for step, (x, y) in enumerate(training_dataloader_list[extra_domain]):
                    score = self.mlp_dispatch.specific_forward(x, did)
                    loss = nn.functional.cross_entropy(score, y)

                    self.mlp_dispatch.specific_optimizor_list[did].zero_grad()
                    loss.backward()
                    self.mlp_dispatch.specific_optimizor_list[did].step()
                    exp_loss += loss.detach().cpu()
                    cnt += 1
                current_params = self.mlp_dispatch.specific_mlp_list[did].state_dict()
                for name, param in saved_params.items():
                    saved_params[name] = (1 - self.params['gamma']) * saved_params[name] + self.params['gamma'] * \
                                         current_params[name]
                self.mlp_dispatch.specific_mlp_list[did].load_state_dict(saved_params)

            exp_loss /= cnt

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
