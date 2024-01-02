import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset


class Negative_sampler():
    def __init__(self, num_domain):
        self.domain_data_list = [[] for i in range(num_domain)]
        self.num_domain = num_domain

    def append(self, trajectory, did):
        self.domain_data_list[did].append(trajectory)

    def negative_sample_trajectory(self, domain_id):
        except_id = domain_id
        while except_id == domain_id:
            except_id = random.randint(0, self.num_domain - 1)
        return random.choice(self.domain_data_list[except_id])

    def negative_sample_interactions(self, domain_id):
        except_id = domain_id
        while except_id == domain_id:
            except_id = random.randint(0, self.num_domain - 1)
        return random.choice(random.choice(self.domain_data_list[except_id]))


class Batch_iterator():
    def __init__(self, batch_size, tensor_list, shuffle=True):
        self.batch_size = batch_size
        self.tensor_list = tensor_list
        self.length = len(tensor_list)
        self.shuffle = shuffle
        self.now = 0
        if self.length < self.batch_size:
            raise Exception('Error')

    def next(self):
        """
        Get next batch of data and move the point forward.
        :return: next batch of (user, item, inter)
        """
        st = self.now
        ed = self.now + self.batch_size
        if ed < self.length:
            self.now += self.batch_size
            return self.tensor_list[st:ed]
        else:
            self.now = (self.now + self.batch_size) % self.batch_size
            part_before = self.tensor_list[st:]
            if self.shuffle:
                random.shuffle(self.tensor_list)
            return part_before + self.tensor_list[:self.now]

    def get_max_step(self):
        return int(self.length / self.batch_size)


class MoE_enhance_adaptor(nn.Module):
    def __init__(self, config, params):
        super(MoE_enhance_adaptor, self).__init__()

        self.input_dim = config['user_dim'] + config['item_dim']
        self.moe_embedding_size = params['moe_embedding_size']
        self.moe_k = params['moe_k']

        self.b = nn.Parameter(torch.randn(size=(self.moe_k, self.input_dim)))
        self.w1 = nn.Parameter(torch.randn(size=(self.moe_k, self.input_dim, self.moe_embedding_size)))
        self.w2 = nn.Parameter(torch.randn(size=(self.input_dim, self.moe_k)))
        self.w3 = nn.Parameter(torch.randn(size=(self.input_dim, self.moe_k)))

    def forward(self, x, train, noise_epsilon=1e-2):
        x = x.unsqueeze(-2)
        tilde_x = (x - self.b)
        tilde_x = tilde_x.unsqueeze(-2)
        tilde_x = torch.matmul(tilde_x, self.w1).squeeze()

        if train:
            delta = torch.nn.functional.softplus(torch.matmul(x, self.w3).squeeze() + noise_epsilon)
            delta = torch.randn_like(delta) * delta
        else:
            delta = 0.0

        g = torch.softmax((torch.matmul(x, self.w2).squeeze() + delta), dim=-1).unsqueeze(-1)
        embedding = torch.sum(g * tilde_x, dim=-2)

        return embedding


class Multi_head_sequence_embedding(nn.Module):
    def __init__(self, config, params):
        super(Multi_head_sequence_embedding, self).__init__()

        self.moe_embedding_size = params['moe_embedding_size']

        self.query_layer = nn.Linear(self.moe_embedding_size, self.moe_embedding_size)
        self.key_layer = nn.Linear(self.moe_embedding_size, self.moe_embedding_size)
        self.value_layer = nn.Linear(self.moe_embedding_size, self.moe_embedding_size)

    def forward(self, traj):
        querys = self.query_layer(traj).unsqueeze(-2)
        keys = self.key_layer(traj).unsqueeze(-3)
        values = self.value_layer(traj)

        attn = torch.softmax(torch.sum(querys * keys, dim=-1), dim=-1).unsqueeze(-1)
        vectors = torch.sum(attn * values, dim=-2)

        return vectors


class FNN_aggregation(nn.Module):
    def __init__(self, config, params):
        super(FNN_aggregation, self).__init__()

        self.moe_embedding_size = params['moe_embedding_size']

        self.hidden_layer = nn.Linear(self.moe_embedding_size, self.moe_embedding_size)
        self.output_layer = nn.Linear(self.moe_embedding_size, self.moe_embedding_size)

    def forward(self, traj):
        x = torch.relu(self.hidden_layer(traj))
        x = self.output_layer(x)
        x = torch.mean(x, dim=-2)
        return x


class UniSRec(nn.Module):
    def __init__(self, config, params, user_profile, item_profile):
        super(UniSRec, self).__init__()

        self.config = config
        self.params = params
        self.device = config['device']
        self.batch_size = params['batch_size']
        self.epoch = params['epoch']
        self.lr = params['lr']

        self.moe_enhance_adaptor = MoE_enhance_adaptor(config, params)
        self.sequence_embedding = Multi_head_sequence_embedding(config, params)
        self.fnn_aggregation = FNN_aggregation(config, params)

        self.user_profile = user_profile
        self.item_profile = item_profile

        self.optimizor = torch.optim.SGD(self.parameters(), lr=self.params['lr'])

    def forward(self, u, v):
        # print(u.shape,v.shape)
        # print(u.shape)
        if len(u.shape) == 2:
            u_slice = u[:, :self.l]
        else:
            u_slice = u[:self.l]
        # raise
        s = torch.concat([u, u_slice], dim=-1)
        t = torch.concat([u, v], dim=-1)
        traj = torch.stack([s, t], dim=-2)
        traj = self.moe_enhance_adaptor.forward(traj, train=False)
        vectors = self.sequence_embedding(traj)
        sequence = self.fnn_aggregation(vectors)
        score = torch.softmax(torch.sum(sequence * vectors, dim=-1), dim=-1).unsqueeze(-2)

        return score

    def train(self, training_domains, valiation_domain):
        negative_sampler = Negative_sampler(len(training_domains))
        user_trajectory_dict = {}
        for did, d in enumerate(training_domains):
            for uid, iid, label in d:
                if label == 0:
                    continue
                if (did, uid) not in user_trajectory_dict:
                    user_trajectory_dict[(did, uid)] = []
                user_trajectory_dict[(did, uid)].append(iid)
        trajectory_dataset = []
        for (did, uid), item_list in user_trajectory_dict.items():
            user_traj_tensor = []
            user_profile = self.user_profile[uid]
            for iid in item_list:
                item_profile = self.item_profile[iid]
                self.l = len(item_profile)
                x = user_profile + item_profile
                user_traj_tensor.append(x)
            user_traj_tensor.append(user_profile + user_profile[:self.l])
            user_traj_tensor = torch.tensor(user_traj_tensor).to(self.device)
            trajectory_dataset.append((did, user_traj_tensor))
            negative_sampler.append(user_traj_tensor, did)

        training_batch_iterator = Batch_iterator(self.batch_size, trajectory_dataset)

        u_list, v_list, y_list = [], [], []
        for uid, iid, label in valiation_domain:
            u = self.user_profile[uid]
            v = self.item_profile[iid]
            y = label
            u_list.append(u)
            v_list.append(v)
            y_list.append(y)
        u_list = torch.tensor(u_list).to(self.device)
        v_list = torch.tensor(v_list).to(self.device)
        y_list = torch.tensor(y_list).to(self.device)
        validation_dataloader = DataLoader(
            TensorDataset(u_list, v_list, y_list),
            batch_size=1,
            shuffle=False
        )

        best_param = self.state_dict()
        min_valid_loss = torch.tensor(100000.0)

        fail_ind = 0
        for e in range(self.epoch):
            exp_loss = torch.tensor(0.0)
            nit = 0

            for step in range(training_batch_iterator.get_max_step()):
                batch_tensor_list = training_batch_iterator.next()
                for (did, traj) in batch_tensor_list:
                    traj = self.moe_enhance_adaptor.forward(traj, train=True)
                    vectors = self.sequence_embedding(traj)
                    sequence = self.fnn_aggregation(vectors)

                    sequence_list = []
                    for ind in range(vectors.shape[0]):
                        sequence_list.append(self.fnn_aggregation(vectors[:ind + 1]))
                    sequence_list = torch.stack(sequence_list, dim=-2)
                    sequence_list = torch.softmax(torch.sum(sequence_list * vectors, dim=-1) / self.params['tau'],
                                                  dim=-1)
                    seq2vec_loss = min(-torch.sum(torch.log2(sequence_list), dim=-1), 10.0)  # clip

                    negative_traj = negative_sampler.negative_sample_trajectory(did)
                    negative_traj = self.moe_enhance_adaptor.forward(negative_traj, train=True)
                    negative_vectors = self.sequence_embedding(negative_traj)
                    sequence_aug = self.fnn_aggregation(vectors[:random.randint(1, vectors.shape[0])])
                    sequence_neg = self.fnn_aggregation(negative_vectors)

                    pos_score = torch.sum(sequence * sequence_aug / self.params['tau'], dim=-1)
                    neg_score = torch.sum(sequence * sequence_neg / self.params['tau'], dim=-1)
                    scores = torch.softmax(torch.stack([pos_score, neg_score]), dim=-1)

                    seq2seq_loss = min(-torch.log2(scores[0]), 10)

                    loss = seq2vec_loss + self.params['lambda'] * seq2seq_loss

                    if type(loss) == float:
                        continue

                    self.optimizor.zero_grad()
                    loss.backward()
                    self.optimizor.step()
                    exp_loss += loss.detach().cpu()
                    nit += 1

            exp_loss /= (nit + 1)

            valid_loss = torch.tensor(0.0)
            for step, (u, v, y) in enumerate(validation_dataloader):
                score = self.forward(u, v)
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
        score = []
        for i in range(user_profile.shape[0]):
            score.append(self.forward(user_profile[i].to(self.device), item_profile[i].to(self.device))[:, 1])
        return torch.concat(score, dim=-1).cpu()
