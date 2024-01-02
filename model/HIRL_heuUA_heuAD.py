from copy import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans


class Invariant_component(nn.Module):
    def __init__(self, config, params):
        super(Invariant_component, self).__init__()

        self.input_dim = config['user_dim'] + config['item_dim']
        self.mask1_hidden_dim = params['mask1_hidden_dim']
        self.mask2_hidden_dim = params['mask2_hidden_dim']
        self.embedding_dim = params['embedding_dim']
        self.predictor_hidden_dim1 = params['predictor_hidden_dim1']
        self.predictor_hidden_dim2 = params['predictor_hidden_dim2']

        # 1 Embedding layer parameters: if the profile is one-hot, we should change it to nn.Embedding().
        self.embedding_layer = nn.Linear(self.input_dim, self.embedding_dim)

        # 2 Environment information mask parameters
        self.mask1_hidden_layer = nn.Linear(self.embedding_dim, self.mask1_hidden_dim)
        self.mask1_output_layer = nn.Linear(self.mask1_hidden_dim, self.embedding_dim)

        # 3 Domain mask information parameters
        self.mask2_hidden_layer = nn.Linear(self.embedding_dim, self.mask2_hidden_dim)
        self.mask2_output_layer = nn.Linear(self.mask2_hidden_dim, self.embedding_dim)

        # 4 Environment information reconstruct parameters
        self.reconstruct1_layer = nn.Linear(self.embedding_dim * 2, self.embedding_dim)

        # 5 Domain information reconstruct parameters
        self.reconstruct2_layer = nn.Linear(self.embedding_dim * 2, self.embedding_dim)

        # 6 Predict function parameters
        self.predict_hidden_layer1 = nn.Linear(self.embedding_dim, self.predictor_hidden_dim1)
        self.predict_hidden_layer2 = nn.Linear(self.predictor_hidden_dim1, self.predictor_hidden_dim2)
        self.predict_output_layer = nn.Linear(self.predictor_hidden_dim2, 2)

    def embedding(self, x):
        return self.embedding_layer(x)

    def environment_mask(self, embedding):
        x = torch.tanh(self.mask1_hidden_layer(embedding))
        x = self.mask1_output_layer(x)
        return torch.sigmoid(x)

    def environment_inv_embedding(self, embedding, environment_mask):
        return torch.multiply(embedding, environment_mask)

    def environment_var_embedding(self, embedding, environment_mask):
        return torch.multiply(embedding, (1 - environment_mask))

    def domain_mask(self, environment_inv_embedding):
        x = torch.tanh(self.mask2_hidden_layer(environment_inv_embedding))
        x = self.mask2_output_layer(x)
        return torch.sigmoid(x)

    def domain_inv_embedding(self, environment_inv_embedding, domain_mask):
        return torch.multiply(environment_inv_embedding, domain_mask)

    def domain_var_embedding(self, environment_inv_embedding, domain_mask):
        return torch.multiply(environment_inv_embedding, (1 - domain_mask))

    def environment_reconstruct_embedding(self, environment_inv_embedding, environment_var_embedding):
        return self.reconstruct1_layer(torch.cat((environment_inv_embedding, environment_var_embedding), dim=-1))

    def domain_reconstruct_embedding(self, domain_inv_embedding, domain_var_embedding):
        return self.reconstruct2_layer(torch.cat((domain_inv_embedding, domain_var_embedding), dim=-1))

    def predict(self, domain_inv_embedding):
        x = torch.tanh(self.predict_hidden_layer1(domain_inv_embedding))
        x = torch.tanh(self.predict_hidden_layer2(x))
        x = self.predict_output_layer(x)
        return x


class Classifier_component(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num):
        super(Classifier_component, self).__init__()

        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, class_num)

    def forward(self, x):
        x = torch.tanh(self.hidden_layer(x))
        x = self.output_layer(x)
        return x


class Classifier_Box(nn.Module):
    def __init__(self, embedding_dim, config, params):
        super(Classifier_Box, self).__init__()

        self.classifier1_component = Classifier_component(embedding_dim, params['classifier1_hidden_dim'],
                                                          config['num_source_domain'] - 1)
        self.classifier2_component = Classifier_component(embedding_dim, params['classifier2_hidden_dim'],
                                                          config['num_source_domain'])


class HIRL_heuUA_heuAD(nn.Module):
    def __init__(self, config, params, user_profile, item_profile):
        super(HIRL_heuUA_heuAD, self).__init__()

        self.params = params
        self.device = config['device']

        self.input_dim = config['user_dim'] + config['item_dim']
        self.embedding_dim = params['embedding_dim']
        self.batch_size = params['batch_size']
        self.epoch = params['epoch']

        self.invariant_component = Invariant_component(config, params)
        self.classifier_box = Classifier_Box(self.embedding_dim, config, params)
        self.cluster_center = []  # Will obtain during initialized training process

        self.user_profile = user_profile
        self.item_profile = item_profile

        self.invariant_optimizor = opt.SGD(self.invariant_component.parameters(), lr=params['invariant_lr'])
        self.classifier_optimizor = opt.SGD(self.classifier_box.parameters(), lr=params['classifier_lr'])

        self.saved_invariant_component_params = self.invariant_component.state_dict()

    def forward(self, x):
        embedding = self.invariant_component.embedding(x)
        environment_mask = self.invariant_component.environment_mask(embedding)
        environment_inv_embedding = self.invariant_component.environment_inv_embedding(embedding, environment_mask)
        domain_mask = self.invariant_component.domain_mask(environment_inv_embedding)
        domain_inv_embedding = self.invariant_component.domain_inv_embedding(environment_inv_embedding, domain_mask)
        score = self.invariant_component.predict(domain_inv_embedding)
        return score

    def predict(self, user_profile, item_profile):
        x = torch.cat([user_profile, item_profile], dim=1).to(self.device)
        return torch.softmax(self.forward(x), dim=1)[:, 1].cpu()

    def train(self, training_domains, valiation_domain):
        # In this section, we want to obtain the fixed cluster-center for each domain.
        # And calculate the environment prob. of each interaction.
        training_set = []
        for did, d in enumerate(training_domains):
            # Deal with each domain d
            x_list_cluster = []
            for (uid, iid, y) in d:
                x = self.user_profile[uid] + self.item_profile[iid]
                x_list_cluster.append(x)
            x_list_cluster = torch.tensor(x_list_cluster)
            cluster = KMeans(n_clusters=len(training_domains))
            cluster.fit(x_list_cluster)
            x_centers = cluster.cluster_centers_
            self.cluster_center.append(x_centers.tolist())
            x_list_cluster = torch.unsqueeze(x_list_cluster, dim=1)
            # print(x_list_cluster.shape, x_centers.shape)
            x_prob = torch.softmax(-torch.norm(x_list_cluster - x_centers, dim=-1), dim=-1).detach().tolist()

            for ind, (uid, iid, y) in enumerate(d):
                training_set.append((uid, iid, y, did, x_prob[ind]))

        x_list, y_list, d_list, p_list = [], [], [], []
        for uid, iid, label, did, prob in training_set:
            x = self.user_profile[uid] + self.item_profile[iid]
            y = label
            d = did
            p = prob
            x_list.append(x)
            y_list.append(y)
            d_list.append(d)
            p_list.append(p)
        x_list = torch.tensor(x_list).to(self.device)
        y_list = torch.tensor(y_list).to(self.device)
        d_list = torch.tensor(d_list).to(self.device)
        p_list = torch.tensor(p_list).to(self.device)
        training_dataloader = DataLoader(
            TensorDataset(x_list, y_list, d_list, p_list),
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
        self.cluster_center = torch.tensor(self.cluster_center)

        best_param = self.state_dict()
        min_valid_loss = torch.tensor(100000.0)

        fail_ind = 0
        for e in range(self.epoch):
            training_predict_loss = torch.tensor(0.0)
            stage1_exp_loss = torch.tensor(0.0)
            stage2_exp_loss = torch.tensor(0.0)
            for step, (x, y, d, p) in enumerate(training_dataloader):
                # 1 Optimize Invariant Component
                embedding = self.invariant_component.embedding(x)

                # - Environment-level information
                environment_mask = self.invariant_component.environment_mask(embedding)
                environment_inv_embedding = self.invariant_component.environment_inv_embedding(embedding,
                                                                                               environment_mask)
                environment_var_embedding = self.invariant_component.environment_var_embedding(embedding,
                                                                                               environment_mask)

                # - Domain-level information
                domain_mask = self.invariant_component.domain_mask(environment_inv_embedding)
                domain_inv_embedding = self.invariant_component.domain_inv_embedding(environment_inv_embedding,
                                                                                     domain_mask)
                domain_var_embedding = self.invariant_component.domain_var_embedding(environment_inv_embedding,
                                                                                     domain_mask)

                # - Reconstruction information
                environment_reconstruct_embedding = self.invariant_component.environment_reconstruct_embedding(
                    environment_inv_embedding, environment_var_embedding)
                domain_reconstruct_embedding = self.invariant_component.domain_reconstruct_embedding(
                    domain_inv_embedding, domain_var_embedding)

                # - Prediction score
                score = self.invariant_component.predict(domain_inv_embedding)

                # - Loss
                loss_predict = torch.nn.functional.cross_entropy(score, y)
                loss_environment_reconstruct = torch.nn.functional.mse_loss(environment_reconstruct_embedding,
                                                                            embedding)
                loss_domain_reconstruct = torch.nn.functional.mse_loss(domain_reconstruct_embedding,
                                                                       domain_inv_embedding)
                loss_environment_classify = torch.nn.functional.mse_loss(p, torch.softmax(
                    self.classifier_box.classifier1_component(environment_var_embedding), dim=-1)) - \
                                            torch.nn.functional.mse_loss(p, torch.softmax(
                                                self.classifier_box.classifier1_component(environment_inv_embedding),
                                                dim=-1))
                loss_domain_classify = torch.nn.functional.cross_entropy(
                    self.classifier_box.classifier2_component(domain_var_embedding), d) - \
                                       torch.nn.functional.cross_entropy(
                                           self.classifier_box.classifier2_component(domain_inv_embedding), d)

                loss = loss_predict + \
                       self.params['invariant_loss_alpha1'] * loss_environment_reconstruct + self.params[
                           'invariant_loss_alpha2'] * loss_domain_reconstruct + \
                       self.params['invariant_loss_alpha3'] * loss_environment_classify + self.params[
                           'invariant_loss_alpha4'] * loss_domain_classify

                self.invariant_optimizor.zero_grad()
                loss.backward()
                self.invariant_optimizor.step()

                training_predict_loss += loss_predict.detach().cpu()
                stage1_exp_loss += loss.detach().cpu()

                # 2 Optimize Classifier
                embedding = self.invariant_component.embedding(x)

                # - Environment-level information
                environment_mask = self.invariant_component.environment_mask(embedding)
                environment_inv_embedding = self.invariant_component.environment_inv_embedding(embedding,
                                                                                               environment_mask)
                environment_var_embedding = self.invariant_component.environment_var_embedding(embedding,
                                                                                               environment_mask)

                # - Domain-level information
                domain_mask = self.invariant_component.domain_mask(environment_inv_embedding)
                domain_inv_embedding = self.invariant_component.domain_inv_embedding(environment_inv_embedding,
                                                                                     domain_mask)
                domain_var_embedding = self.invariant_component.domain_var_embedding(environment_inv_embedding,
                                                                                     domain_mask)

                # - Prediction score

                loss_environment = torch.nn.functional.mse_loss(p, torch.softmax(
                    self.classifier_box.classifier1_component(environment_var_embedding), dim=-1)) + \
                                   torch.nn.functional.mse_loss(p, torch.softmax(
                                       self.classifier_box.classifier1_component(environment_inv_embedding), dim=-1))
                loss_domain = torch.nn.functional.cross_entropy(
                    self.classifier_box.classifier2_component(domain_var_embedding), d) + \
                              torch.nn.functional.cross_entropy(
                                  self.classifier_box.classifier2_component(domain_inv_embedding), d)

                loss = loss_environment + self.params['classifier_lr'] * loss_domain

                self.classifier_optimizor.zero_grad()
                loss.backward()
                self.classifier_optimizor.step()

                stage2_exp_loss += loss.detach().cpu()

                # 4 Environment Refinement

                valid_loss_before = torch.tensor(0.0)
                # Calculate validation loss before refinement
                for step_valid, (x_valid, y_valid) in enumerate(validation_dataloader):
                    score_valid = self.forward(x_valid)
                    loss_valid = nn.functional.cross_entropy(score_valid, y_valid)
                    valid_loss_before += loss_valid.detach().cpu()
                    break

                # Save the params
                self.saved_invariant_component_params = self.invariant_component.state_dict()

                # Addition Disturbance
                entropy = torch.sum(torch.multiply(p, torch.log2(p)), dim=-1)
                val, ind = torch.topk(entropy, k=int(0.5 * entropy.shape[0]))

                # Obtain selected x.
                x_selected = x[ind]
                y_selected = y[ind]
                d_selected = d[ind]
                p_selected = p[ind]

                # Optimize selected samples for invariant components.
                embedding = self.invariant_component.embedding(x_selected)

                # - Environment-level information
                environment_mask = self.invariant_component.environment_mask(embedding)
                environment_inv_embedding = self.invariant_component.environment_inv_embedding(embedding,
                                                                                               environment_mask)
                environment_var_embedding = self.invariant_component.environment_var_embedding(embedding,
                                                                                               environment_mask)

                # - Domain-level information
                domain_mask = self.invariant_component.domain_mask(environment_inv_embedding)
                domain_inv_embedding = self.invariant_component.domain_inv_embedding(environment_inv_embedding,
                                                                                     domain_mask)
                domain_var_embedding = self.invariant_component.domain_var_embedding(environment_inv_embedding,
                                                                                     domain_mask)

                # - Reconstruction information
                environment_reconstruct_embedding = self.invariant_component.environment_reconstruct_embedding(
                    environment_inv_embedding, environment_var_embedding)
                domain_reconstruct_embedding = self.invariant_component.domain_reconstruct_embedding(
                    domain_inv_embedding, domain_var_embedding)

                # - Prediction score
                score = self.invariant_component.predict(domain_inv_embedding)

                # - Loss
                loss_predict = torch.nn.functional.cross_entropy(score, y_selected)
                loss_environment_reconstruct = torch.nn.functional.mse_loss(environment_reconstruct_embedding,
                                                                            embedding)
                loss_domain_reconstruct = torch.nn.functional.mse_loss(domain_reconstruct_embedding,
                                                                       domain_inv_embedding)
                loss_environment_classify = torch.nn.functional.mse_loss(p_selected,
                                                                         torch.softmax(
                                                                             self.classifier_box.classifier1_component(
                                                                                 environment_var_embedding), dim=-1)) - \
                                            torch.nn.functional.mse_loss(p_selected,
                                                                         torch.softmax(
                                                                             self.classifier_box.classifier1_component(
                                                                                 environment_inv_embedding), dim=-1))
                loss_domain_classify = torch.nn.functional.cross_entropy(
                    self.classifier_box.classifier2_component(domain_var_embedding), d_selected) - \
                                       torch.nn.functional.cross_entropy(
                                           self.classifier_box.classifier2_component(domain_inv_embedding), d_selected)

                loss = loss_predict + \
                       self.params['invariant_loss_alpha1'] * loss_environment_reconstruct + self.params[
                           'invariant_loss_alpha2'] * loss_domain_reconstruct + \
                       self.params['invariant_loss_alpha3'] * loss_environment_classify + self.params[
                           'invariant_loss_alpha4'] * loss_domain_classify

                self.invariant_optimizor.zero_grad()
                loss.backward()
                self.invariant_optimizor.step()

                # Calculate validation loss after refinement
                valid_loss_after = torch.tensor(0.0)
                for step_valid, (x_valid, y_valid) in enumerate(validation_dataloader):
                    score_valid = self.forward(x_valid)
                    loss_valid = nn.functional.cross_entropy(score_valid, y_valid)
                    valid_loss_after += loss_valid.detach().cpu()
                    break

                if valid_loss_before < valid_loss_after:
                    self.invariant_component.load_state_dict(self.saved_invariant_component_params)

                # print('Valid_loss(before): %f.  Valid_loss(after): %f.' %(valid_loss_before,valid_loss_after))

            training_predict_loss /= (step + 1)
            stage1_exp_loss /= (step + 1)
            stage2_exp_loss /= (step + 1)

            print('--[Epoch %d]--' % e)
            print('Training prediction loss: %f.' % training_predict_loss)
            print('(Stage 1) Invariant learning experience loss: %f.' % stage1_exp_loss)
            print('(Stage 2) Classifier adversarial experience loss: %f.' % stage2_exp_loss)

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

            print('Valid loss: %f, min valid loss: %f.' % (valid_loss, min_valid_loss))

            if fail_ind >= 10:
                print('Early Stop with 10 steps.')
                self.load_state_dict(best_param)
                return min_valid_loss.numpy()
        self.load_state_dict(best_param)
        return min_valid_loss.numpy()
