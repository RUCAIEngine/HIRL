import copy
import random

import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def cal_h_loss(dim: int, adj_matrix: torch.Tensor) -> torch.Tensor:
    cof = 1.0
    dim_float = float(dim)
    device = adj_matrix.device
    z: torch.Tensor = adj_matrix * adj_matrix
    z_in: torch.Tensor = torch.eye(dim).to(device)
    dag_loss: torch.Tensor = torch.tensor(dim_float).to(device)
    for i in range(1, dim + 1):
        z_in = torch.mm(z_in, z)

        # print(z_in.cpu())
        dag_loss += (1.0 / cof) * z_in.trace()
        cof *= float(i)

        if float(torch.mean(z_in).cpu()) > 90000.:
            # print('!!!!')
            break

    dag_loss -= dim_float
    return dag_loss


def BPR_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    # loss = torch.mean(softplus(neg_scores - pos_scores))
    # loss = - torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
    loss = - torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
    return loss


class LinearCG(nn.Module):
    def __init__(self, user_dim: int, item_dim: int):
        super(LinearCG, self).__init__()

        self.user_dim = user_dim
        self.item_dim = item_dim
        self.adj_dim = item_dim
        self.adj_matrix: torch.nn.Parameter = torch.nn.Parameter(torch.rand((self.item_dim, self.item_dim)),
                                                                 requires_grad=True)
        self.user2item: torch.nn.Sequential = torch.nn.Sequential(
            nn.Linear(self.user_dim, self.item_dim, bias=False)
        )
        self.__init_weight()

    def __init_weight(self):
        nn.init.normal_(self.adj_matrix, std=0.15)

    def forward(self, user_samples: torch.Tensor, item_samples: torch.Tensor,
                need_preference_grad: bool = True):
        assert user_samples.shape[0] == item_samples.shape[0]

        if need_preference_grad:
            user2item_causal_graph: torch.nn.Sequential = self.user2item
            item_causal_graph: torch.nn.Parameter = self.adj_matrix
        else:
            user2item_causal_graph: torch.nn.Sequential = copy.deepcopy(self.user2item)
            item_causal_graph: torch.nn.Parameter = self.adj_matrix.detach()

        user2item_vector: torch.Tensor = user2item_causal_graph(user_samples)

        item2item_vector: torch.Tensor = torch.mm(item_samples, item_causal_graph)

        user_preference: torch.Tensor = user2item_vector + item2item_vector
        return user_preference

    def get_numpy_adj_matrix(self, threshold: float = 0.3,
                             need_threshold: bool = True, print_weight: bool = True) -> np.ndarray:
        w_est: np.ndarray = self.adj_matrix.cpu().detach().numpy().copy()
        if not need_threshold:
            return w_est
        w_est[np.abs(w_est) < threshold] = 0
        return w_est

    def get_tensor_adj_matrix(self) -> torch.Tensor:
        """
        :return: adj matrix. CAREFUL! this tensor is in the device where the model be. so you should check the device!
        """
        return self.adj_matrix


class CausPref(nn.Module):
    def __init__(self, config, params, user_profile, item_profile):
        super(CausPref, self).__init__()

        self.params = params
        self.device = config['device']
        self.user_dim = config['user_dim']
        self.item_dim = config['item_dim']
        self.user_pref_dim = config['item_dim']
        self.latent_dim = params['latent_dim']
        self.batch_size = params['batch_size']
        self.epoch = params['epoch']
        self.lr = params['lr']

        self.causal_graph = LinearCG(self.user_dim, self.item_dim)

        self.user_encoder: nn.Sequential = nn.Sequential(
            nn.Linear(self.user_pref_dim, self.latent_dim, bias=True),
        )

        self.item_encoder: nn.Sequential = nn.Sequential(
            nn.Linear(self.item_dim, self.latent_dim, bias=True),
        )

        self.user_profile = user_profile
        self.item_profile = item_profile

        self.optimizor = torch.optim.Adam(self.parameters(), lr=params['lr'])

    def get_numpy_adj_matrix(self, threshold: float = 0.3,
                             need_threshold: bool = True, print_weight: bool = True) -> np.ndarray:
        return self.causal_graph.get_numpy_adj_matrix(threshold, need_threshold, print_weight)

    def get_tensor_adj_matrix(self) -> torch.Tensor:
        """
        :return: adj matrix. CAREFUL! this tensor is in the device where the model be. so you should check the device!
        """
        return self.causal_graph.get_tensor_adj_matrix()

    def forward(self, user_samples: torch.Tensor, item_samples: torch.Tensor,
                need_preference_grad: bool = True, need_user_pref: bool = False):
        assert user_samples.shape[0] == item_samples.shape[0]

        user_pref: torch.Tensor = self.causal_graph(user_samples, item_samples, need_preference_grad)

        user_embedding: torch.Tensor = self.user_encoder(user_pref)

        item_embedding: torch.Tensor = self.item_encoder(item_samples)

        if need_user_pref:
            return torch.sum(user_embedding * item_embedding, dim=1), user_pref
        else:
            return torch.sum(user_embedding * item_embedding, dim=1)

    def train(self, training_domains, valiation_domain):
        training_set = set()
        for d in training_domains:
            training_set |= d
        positive_user_list, positive_item_list, negative_user_list, negative_item_list = [], [], [], []
        for uid, iid, label in training_set:
            u = self.user_profile[uid]
            v = self.item_profile[iid]
            y = label
            if y == 1:
                positive_user_list.append(u)
                positive_item_list.append(v)
            else:
                negative_user_list.append(u)
                negative_item_list.append(v)
        positive_user_list = torch.tensor(positive_user_list).to(self.device)
        positive_item_list = torch.tensor(positive_item_list).to(self.device)
        ind_box = np.random.choice(range(len(negative_user_list)), size=positive_user_list.shape[0], replace=False)
        negative_user_list = torch.tensor(negative_user_list)[ind_box].to(self.device)
        negative_item_list = torch.tensor(negative_item_list)[ind_box].to(self.device)
        training_dataloader = DataLoader(
            TensorDataset(positive_user_list, positive_item_list, negative_user_list, negative_item_list),
            batch_size=self.batch_size,
            shuffle=True
        )

        user_list, item_list, y_list = [], [], []
        for uid, iid, label in valiation_domain:
            user = self.user_profile[uid]
            item = self.item_profile[iid]
            y = label
            user_list.append(user)
            item_list.append(item)
            y_list.append(y)
        user_list = torch.tensor(user_list).to(self.device)
        item_list = torch.tensor(item_list).to(self.device)
        y_list = torch.tensor(y_list).to(self.device)
        validation_dataloader = DataLoader(
            TensorDataset(user_list, item_list, y_list),
            batch_size=self.batch_size,
            shuffle=False
        )

        best_param = self.state_dict()
        min_valid_loss = torch.tensor(100000.0)

        fail_ind = 0
        need_preference_grad = True
        for e in range(self.epoch):
            exp_loss = torch.tensor(0.0)
            for step, (positive_user, positive_item, negative_user, negative_item) in enumerate(training_dataloader):
                mse = nn.MSELoss()
                pos_scores, rec = self.forward(positive_user,
                                               positive_item, need_user_pref=True)
                neg_scores = self.forward(negative_user, negative_item,
                                          need_preference_grad=need_preference_grad)

                bpr_loss = BPR_loss(pos_scores, neg_scores)

                rec_loss = mse(rec, positive_item)

                adj_matrix = self.get_tensor_adj_matrix()

                h_loss = cal_h_loss(self.item_dim, adj_matrix)

                dag_loss = 0.5 * h_loss * h_loss + self.params['reg_alpha'] * h_loss

                dim_float = float(self.item_dim)
                item_sparse = torch.norm(adj_matrix, p=1) / (dim_float * dim_float)

                u2i_sparse_loss = torch.norm(self.causal_graph.user2item[0].weight, p=1)

                major_loss: torch.Tensor = \
                    rec_loss * self.params['rec_coe'] + self.params['bpr_coe'] * bpr_loss + dag_loss + \
                    u2i_sparse_loss * self.params['u2i_sparse_coe'] + item_sparse * self.params['item_sparse_coe']

                self.optimizor.zero_grad()
                major_loss.backward()
                self.optimizor.step()

                exp_loss += major_loss.detach().cpu()
            exp_loss /= (step + 1)

            valid_loss = torch.tensor(0.0)
            for step, (user, item, y) in enumerate(validation_dataloader):
                score = self.predict_valid(user, item)
                loss = torch.nn.functional.mse_loss(score, y.float())
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

    def predict(self, users_attr: torch.Tensor, items_attr: torch.Tensor):
        assert len(users_attr.shape) == 2 and len(items_attr.shape) == 2

        users_attr = users_attr.to(self.device)
        items_attr = items_attr.to(self.device)

        user_pref_tensor: torch.Tensor = self.causal_graph(users_attr, items_attr)

        user_embedding: torch.Tensor = self.user_encoder(user_pref_tensor)

        item_embedding: torch.Tensor = self.item_encoder(items_attr)
        score = torch.sum(user_embedding * item_embedding, dim=-1)

        return torch.sigmoid(score).cpu()

    def predict_valid(self, users_attr: torch.Tensor, items_attr: torch.Tensor):
        # Difference on cuda.
        assert len(users_attr.shape) == 2 and len(items_attr.shape) == 2

        user_pref_tensor: torch.Tensor = self.causal_graph(users_attr, items_attr)

        user_embedding: torch.Tensor = self.user_encoder(user_pref_tensor)

        item_embedding: torch.Tensor = self.item_encoder(items_attr)
        score = torch.sum(user_embedding * item_embedding, dim=-1)

        return torch.sigmoid(score)
