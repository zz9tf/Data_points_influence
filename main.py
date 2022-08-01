import time
from functools import reduce
import numpy as np
import os
import torch
import argparse
from load_data import load_data
from model.generic_neural_net import Model
import custom_methods

start_time = time.time()
print(time.strftime('%H:%M:%S', time.localtime()))

def secondsToStr(t):
    return '%d:%02d:%02d.%03d' % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])

def set_configs():
    parser = argparse.ArgumentParser()
    # detaset
    parser.add_argument('--dataset', type=str, default='fraud_detection',
                        help='name of dataset: movielens, yelp, census_income, churn, fraud_detection')
    parser.add_argument('--datapath', type=str, default='data',
                        help='the path of datasets')
    # basic configs
    parser.add_argument('--model', type=str, default='lr',
                        help='model type: MF or NCF or lr')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='the path of checkpoints storage')
    # train configs
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='the batch_size for training or predict, None for not to use batch')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate for the training model')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='l2 regularization term for the training model')
    # train process configs
    parser.add_argument('--num_epoch_train', type=int, default=30000,
                        help='training steps for the training model')
    parser.add_argument('--load_checkpoint', type=bool, default=False,
                        help='whether loading previous model if it exists')
    parser.add_argument('--save_checkpoint', type=bool, default=False,
                        help='whether saving the current model')
    parser.add_argument('--plot', type=bool, default=False,
                        help='if plot the figure of train loss and test loss')
    # experiment
    parser.add_argument('--experiment', type=str, default='experiment_small_model_select_points',
                        help='the target exprience to execute')
    parser.add_argument('--task_num', type=int, default=1,
                        help='tell experiment method which task part to execute')
    parser.add_argument('--aid_dir', type=str, default='aid',
                        help='the path for aid data needed in expriments')
    parser.add_argument('--experiment_save_dir', type=str, default='experiment_save_results',
                        help='the path for experiments to save result')

    return parser.parse_args()

configs = set_configs()

dataset = load_data(os.path.join(configs.datapath, configs.dataset))
        
model_configs = {
        'MF': {
                'num_users': int(np.max(dataset['train'].x[:, 0]) + 1),
                'num_items': int(np.max(dataset['train'].x[:, 1]) + 1),
                'embedding_size': 16,
                'weight_decay': 1e-2 # l2 regularization term for training MF or NCF model
            },
        'NCF' : {
                'num_users': int(np.max(dataset['train'].x[:, 0]) + 1),
                'num_items': int(np.max(dataset['train'].x[:, 1]) + 1),
                'embedding_size': 16,
                'weight_decay': 1e-2 # l2 regularization term for training MF or NCF model
            },
        'lr' : {
            'input_elem': len(dataset['train'].x[0])
        }
}

model = Model(
    # model
    model_configs=model_configs[configs.model],
    basic_configs={
        'model': configs.model,
        # loading data
        'dataset': dataset,
        # train configs
        'batch_size': configs.batch_size,
        'learning_rate': configs.lr,
        'weight_decay': configs.weight_decay,
        
        # loading configs
        'checkpoint_dir': configs.checkpoint_dir,
        'model_name': '%s_%s_wd%.0e' % (
            configs.dataset, configs.model, configs.weight_decay)
    }
)

if configs.experiment =='test':
    model.train(verbose=True, plot=True)
    eva_x, eva_y = model.np2tensor(model.dataset['test'].get_batch())
    eva_diff = model.model(eva_x) - eva_y
    print(len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff))
    exit()
elif configs.experiment == 'experiment_get_correlation':
    custom_methods.experiment_get_correlation(model=model, configs=configs)
elif configs.experiment == 'experiment_predict_distribution':
    experiment_configs = {
        'precent_to_keep': 0.7
    }
    custom_methods.experiment_predict_distribution(model=model, configs=configs, precent_to_keep=0.7)
elif configs.experiment == 'experiment_possible_higher_accuracy':
    experiment_configs = {
        # all task
        'eva_sets': ['test', 'valid'],
        'num_whole_data_accuracies_task': 1,
        # small_model_performance_task
        'num_small_model_performance_task': 100,
        'remain_percents': [0.1, 0.05]
    }
    custom_methods.experiment_possible_higher_accuracy(model=model, configs=configs, experiment_configs=experiment_configs)
elif configs.experiment == 'experiment_small_model_select_points':
    experiment_configs = {
        # all task
        'performance_eva_sets': ['test', 'valid'],
        'num_whole_data_accuracies_task': 1,
        # rand_model_task
        'num_rand_model_task': 100,
        'rand_remain_percent': 0.1,
        # rand_model_inf_task
        'max_data_points_num_to_calculate': 5000,
        'inf_eva_set': 'test',
        # selected_model
        'num_combined_inf': 1,
        'data_selecting_method': 'mean', # mean or vote
        'select_remain_percent': 0.1,
    }
    custom_methods.experiment_small_model_select_points(model=model, configs=configs, experiment_configs=experiment_configs)
elif configs.experiment == 'experiment_small_model_based_big_model':
    experiment_configs = {
        'performance_eva_sets': ['test', 'valid'],
        # influence value on all data
        'max_data_points_num_to_calculate': 5000,
        'inf_eva_set': 'test',
        # selected small model
        'remain_percent': 0.1
    }
    custom_methods.experiment_small_model_based_big_model(model=model, configs=configs, experiment_configs=experiment_configs)
elif configs.experiment == 'experiment_remove_all_negtive':
    custom_methods.experiment_remove_all_negtive(model=model, configs=configs)
else:
    raise Exception('No such a experiment {}'.format(configs.experiment))

end_time = time.time()
print(time.strftime('%H:%M:%S', time.localtime()))
print('Use time: {}'.format(secondsToStr(end_time - start_time)))