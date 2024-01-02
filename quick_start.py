seed = 1128

import os
from hyper_parameters import *


def get_command(model_name, dataset_name, output_path, hyper_param):
    cmd = 'python run.py '
    cmd += '%s %s %s ' % (model_name, dataset_name, output_path)

    for k, v in hyper_param.items():
        cmd += '%s %s ' % (k, str(v))
    return cmd


def run(model_name, dataset_name, output_path, hyper_param):
    cmd = get_command(model_name, dataset_name, output_path, hyper_param)
    return os.system(cmd)


if __name__ == '__main__':
    os.environ['PYTHONHASHSEED'] = str(seed)
    model_name = 'DNN'
    dataset_tag = 'kuairec_8'

    dataset_name = 'dataset/%s.pickle' % dataset_tag
    output_path = '%s_%s' % (model_name, dataset_tag)
    run(model_name, dataset_name, output_path, eval(model_name + '_dict'))
