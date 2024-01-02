import csv
import importlib
import os
import argparse
import pickle
import random
import time

import numpy as np
import torch

# Env Config
seed = 1128
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Local Config
ns_rate = 4.0
topk = 10
item_candidate_size = 50

# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

print('Current device', device)


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_csv_table(table, path, mode='w'):
    with open(path, mode, newline='') as f:
        writer = csv.writer(f)
        writer.writerows(table)


def check_dir_and_make(path):
    if not os.path.exists(path):
        os.mkdir(path)


def negativeSample(interset):
    itemPopDict = {}

    # Get the popularity dict
    for t in interset:
        if t[1] not in itemPopDict:
            itemPopDict[t[1]] = 1
        else:
            itemPopDict[t[1]] += 1
    itemPopDict = sorted(itemPopDict.items(), key=lambda x: x[1], reverse=True)
    popularItemList = [t[0] for t in itemPopDict]

    # Gather the items on the corresponding user
    userDict = {}
    for t in interset:
        if t[0] not in userDict:
            userDict[t[0]] = [t[1]]
        else:
            userDict[t[0]].append(t[1])
    userDict = sorted(userDict.items(), key=lambda x: len(x[1]), reverse=True)

    # Negative sampling with the ratio ns_rate
    outputset = set()
    for user in userDict:
        uid = user[0]
        tmpset = set(user[1])
        negativeset = set()
        sampleNum = int(ns_rate * len(tmpset))
        for iid in popularItemList:
            if sampleNum == 0:
                break
            if iid not in tmpset:
                negativeset.add(iid)
                sampleNum -= 1
        for iid in tmpset:
            outputset.add((uid, iid, 1))
        for iid in negativeset:
            outputset.add((uid, iid, 0))

    return outputset


def model_train(model, source_domains, model_path):
    source_domains = list(source_domains.values())
    valiation_domain = source_domains[0]
    training_domains = source_domains[1:]

    valiation_domain = negativeSample(valiation_domain)
    training_domains = [negativeSample(d) for d in training_domains]

    valid_loss = model.train(training_domains, valiation_domain)

    torch.save(model, model_path)

    return valid_loss


def ndcg(prediction, target):
    """
    Calculate the ndcg performance
    :param prediction: (np.array) prediction score
    :param target: (np.array) ground truth list
    :return: (float) ndcg score
    """
    length = topk if topk <= len(target) else len(target)
    gd = np.ones(shape=(length,))
    idcg = np.sum(gd * 1. / np.log2(np.arange(2, length + 2)))
    dcg = np.sum(prediction[:length] * 1. / np.log2(np.arange(2, length + 2)))
    ndcg = dcg / idcg
    return ndcg


def recall(prediction, target):
    """
    Calculate the recall performance
    :param prediction: (np.array) prediction score
    :param target: (np.array) ground truth list
    :return: (float) recall score
    """
    TP = np.sum(prediction)
    TPFN = len(target)
    return TP / TPFN


def precision(prediction, target):
    """
    Calculate the precision performance
    :param prediction: (np.array) prediction score
    :param target: (np.array) ground truth list
    :return: (float) precision score
    """
    TP = np.sum(prediction)
    return TP / topk


def f1(prediction, target):
    """
    Calculate the f1 performance
    :param prediction: (np.array) prediction score
    :param target: (np.array) ground truth list
    :return: (float) f1 score
    """
    r = recall(prediction, target)
    p = precision(prediction, target)
    if (r + p) <= 1e-6:
        return 0.0
    return 2 * r * p / (r + p)


def evaluate(model, target_domain, user_profile, item_profile):
    user = target_domain['user']
    item_pool_sample = np.random.choice(list(target_domain['item_pool']), size=item_candidate_size, replace=False)

    inter = target_domain['inter']

    user_gd_dict = {}
    for uid, iid in inter:
        if uid not in user_gd_dict:
            user_gd_dict[uid] = []
        user_gd_dict[uid].append(iid)
    for k, v in user_gd_dict.items():
        user_gd_dict[k] = np.array(v)

    item_pool_base_list = []
    for iid in item_pool_sample:
        item_pool_base_list.append(item_profile[iid])

    item_pool_base_tensor = torch.tensor(item_pool_base_list)
    totalPerformance = {'ndcg': torch.tensor(0.0), 'recall': torch.tensor(0.0), 'f1': torch.tensor(0.0)}

    for uid in user:
        user_p = user_profile[uid]
        user_gd_array = user_gd_dict[uid]
        item_pool_array = np.concatenate([item_pool_sample, user_gd_array])

        user_gd_tensor = []
        for iid in user_gd_array:
            user_gd_tensor.append(item_profile[iid])
        user_gd_tensor = torch.tensor(user_gd_tensor)
        user_sp_gd_tensor = torch.concatenate([item_pool_base_tensor, user_gd_tensor], dim=0)

        user_profile_tensor = torch.tensor(user_p) + torch.zeros(size=(user_sp_gd_tensor.shape[0], len(user_p)))

        score = model.predict(user_profile_tensor, user_sp_gd_tensor)
        top_value, top_indice = torch.topk(score, topk)
        recommend_list = np.array(item_pool_array[top_indice])

        gd_set = set(user_gd_dict[uid])
        pre = list(map(lambda x: x in gd_set, recommend_list))
        r = np.array(pre, dtype=float)

        totalPerformance['ndcg'] += ndcg(r, recommend_list)
        totalPerformance['recall'] += recall(r, recommend_list)
        totalPerformance['f1'] += f1(r, recommend_list)

    for k, v in totalPerformance.items():
        totalPerformance[k] = float((v / len(user)).detach())

    print(totalPerformance)

    return totalPerformance


def model_test(model, target_domain, user_profile, item_profile):
    return evaluate(model, target_domain, user_profile, item_profile)


def run(model_name, dataset_path, output_path, hyper_param):
    check_dir_and_make('param/')
    check_dir_and_make('result/')
    check_dir_and_make('hyper_tune/')
    model_path = 'param/' + model_name + '.model'
    result_path = 'result/' + output_path + '.csv'
    hyper_tune_path = 'hyper_tune/' + output_path + '.csv'

    dataset = read_pickle(dataset_path)
    source_domains = dataset['source_domain']
    target_domains = dataset['target_domain']
    user_profile = dataset['user_profile']
    item_profile = dataset['item_profile']

    config_param = {}
    config_param['user_dim'] = len(list(dataset['user_profile'].values())[0])
    config_param['item_dim'] = len(list(dataset['item_profile'].values())[0])
    config_param['num_source_domain'] = len(source_domains)
    config_param['device'] = device

    # Model Definition
    model = eval('importlib.import_module(\'model\').%s' % model_name)(config_param, hyper_param, user_profile,
                                                                       item_profile)
    model = model.to(device)
    # Model Training
    valid_loss = model_train(model, source_domains, model_path)
    # Model Testing
    performance_list = []
    ndcg_list, recall_list, f1_list = [], [], []

    for tid, target_domain in target_domains.items():
        performance = model_test(model, target_domain, user_profile, item_profile)
        ndcg_list.append(performance['ndcg'])
        recall_list.append(performance['recall'])
        f1_list.append(performance['f1'])
    performance_list.append(
        [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())] + [model_name] + [np.mean(ndcg_list)] + [
            np.mean(recall_list)] + [np.mean(f1_list)])

    # Output performance
    write_csv_table(performance_list, result_path, mode='a')

    # Save validation performance for hyper-parameters tuning
    with open(hyper_tune_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([[hyper_param, valid_loss]])


def convert_hyper_param(extra_args):
    """
    Convert string hyper-parameters into dicts.
    :param extra_args: (str)  hyper-parameters with string form.
    :return: (dict) the dict of hyper-parameters
    """
    hyperParam = {}
    tmpKey, tmpValue = None, None
    for index, ext in enumerate(extra_args):
        if index % 2 == 0:
            tmpKey = ext
        else:
            if ext.isdigit():
                tmpValue = int(ext)
            else:
                tmpValue = float(ext)
            hyperParam[tmpKey] = tmpValue
    return hyperParam


if __name__ == '__main__':
    # Collect model name, dataset and [option] random seed by command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('-seed', default=seed, type=int)
    parser.add_argument('-cuda', default=0, type=int)
    args, extra_args = parser.parse_known_args()
    if args.seed != seed:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    if args.cuda != 0:
        torch.cuda.set_device(args.cuda)

    # Convert string hyper-parameters into dicts.
    hyper_param = convert_hyper_param(extra_args)

    print('------')
    print('Start running:')
    print('Model:', args.model)
    print('Dataset:', args.dataset)
    print('Output_path:', args.output_path)
    print('Hyper-parameters:', hyper_param)

    run(args.model, args.dataset, args.output_path, hyper_param)
