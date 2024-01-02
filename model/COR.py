import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset


class Encoder(nn.Module):
    def __init__(self, config, params):
        super(Encoder, self).__init__()

        self.input_dim = config['user_dim'] + config['item_dim']
        self.hidden_dim = params['encoder_hidden_dim']
        self.output_dim = params['unobs_dim']

        self.hidden_layer_mu = nn.Linear(self.input_dim, self.hidden_dim)
        self.output_layer_mu = nn.Linear(self.hidden_dim, self.output_dim)

        self.hidden_layer_sigma = nn.Linear(self.input_dim, self.hidden_dim)
        self.output_layer_sigma = nn.Linear(self.hidden_dim, self.output_dim)

    def calculate_mu(self, x):
        x = torch.tanh(self.hidden_layer_mu(x))
        return torch.sigmoid(self.output_layer_mu(x))

    def calculate_sigma(self, x):
        x = torch.tanh(self.hidden_layer_sigma(x))
        return torch.sigmoid(self.output_layer_sigma(x))

    def sample_unobs_feature(self, mu, sigma):
        return torch.normal(mu, sigma)


class Decoder(nn.Module):
    def __init__(self, config, params):
        super(Decoder, self).__init__()

        self.user_dim = config['user_dim']
        self.item_dim = config['item_dim']
        self.unobs_dim = params['unobs_dim']

        self.z1_hidden_dim = params['z1_hidden_dim']
        self.z1_dim = params['z1_dim']
        self.z2_hidden_dim = params['z2_hidden_dim']
        self.z2_dim = params['z2_dim']
        self.predict_hidden_dim = params['predict_hidden_dim']

        self.z1_hidden_layer = nn.Linear(self.user_dim + self.unobs_dim, self.z1_hidden_dim)
        self.z1_output_layer = nn.Linear(self.z1_hidden_dim, self.z1_dim)

        self.z2_hidden_layer = nn.Linear(self.unobs_dim, self.z2_hidden_dim)
        self.z2_output_layer = nn.Linear(self.z2_hidden_dim, self.z2_dim)

        self.predict_hidden_layer = nn.Linear(self.z1_dim + self.z2_dim + self.item_dim, self.predict_hidden_dim)
        self.predict_output_layer = nn.Linear(self.predict_hidden_dim, 2)

    def forward(self, obs_feature, unobs_feature, item_feature):
        z1 = torch.concat([obs_feature, unobs_feature], dim=-1)
        z1 = torch.tanh(self.z1_hidden_layer(z1))
        z1 = self.z1_output_layer(z1)

        z2 = torch.tanh(self.z2_hidden_layer(unobs_feature))
        z2 = self.z2_output_layer(z2)

        x = torch.concat([z1, z2, item_feature], dim=-1)
        x = torch.tanh(self.predict_hidden_layer(x))
        x = self.predict_output_layer(x)

        return x


class COR(nn.Module):
    def __init__(self, config, params, user_profile, item_profile):
        super(COR, self).__init__()

        self.config = config
        self.device = config['device']
        self.params = params
        self.batch_size = params['batch_size']
        self.epoch = params['epoch']
        self.lr = params['lr']

        self.encoder = Encoder(config, params)
        self.decoder = Decoder(config, params)

        self.user_profile = user_profile
        self.item_profile = item_profile

        self.optimizor = opt.SGD(self.parameters(), lr=self.lr)

    def forward(self, obs_feature, item_feature):
        x = torch.concat(
            [obs_feature, torch.zeros(size=(obs_feature.shape[0], item_feature.shape[1]), device=self.device)], dim=-1)
        mu = self.encoder.calculate_mu(x)
        sigma = self.encoder.calculate_sigma(x)
        unobs_feature = self.encoder.sample_unobs_feature(mu, sigma)

        score = self.decoder(obs_feature, unobs_feature, item_feature)
        return score

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
                mu = self.encoder.calculate_mu(x)
                sigma = self.encoder.calculate_sigma(x)
                unobs_feature = self.encoder.sample_unobs_feature(mu, sigma)
                obs_feature = x[:, :self.config['user_dim']]
                item_feature = x[:, self.config['user_dim']:]

                score = self.decoder(obs_feature, unobs_feature, item_feature)
                loss1 = nn.functional.cross_entropy(score, y)
                loss2 = torch.mean(- 0.5 * torch.sum(1 + 2 * torch.log2(sigma) - mu * mu - sigma * sigma, dim=-1))
                loss = loss1 + self.params['beta'] * loss2

                self.optimizor.zero_grad()
                loss.backward()
                self.optimizor.step()
                exp_loss += loss.detach().cpu()
            exp_loss /= (step + 1)

            valid_loss = torch.tensor(0.0)
            for step, (x, y) in enumerate(validation_dataloader):
                obs_feature = x[:, :self.config['user_dim']]
                item_feature = x[:, self.config['user_dim']:]
                score = self.forward(obs_feature, item_feature)
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
        return torch.softmax(self.forward(user_profile.to(self.device), item_profile.to(self.device)), dim=1)[:,
               1].cpu()
