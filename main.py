import numpy as np
import os
import torch
import argparse
from load_data import load_data
from model.generic_neural_net import Model
import custom_methods

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
    parser.add_argument('--result_dir', type=str, default='result',
                        help='the path of checkpoints storage')
    # train configs
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='the batch_size for training or predict, None for not to use batch')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate for the training model')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='l2 regularization term for the training model')
    # train process configs
    parser.add_argument('--num_epoch_train', type=int, default=270000,
                        help='training steps for the training model')
    parser.add_argument('--load_checkpoint', type=bool, default=True,
                        help='whether loading previous model if it exists')
    parser.add_argument('--save_checkpoint', type=bool, default=True,
                        help='whether saving the current model')
    parser.add_argument('--plot', type=bool, default=False,
                        help='if plot the figure of train loss and test loss')
    # experiment
    parser.add_argument('--experiment', type=str, default="experiment_small_model_select_points",
                        help='the target exprience to execute')
    parser.add_argument('--task_num', type=int, default=1,
                        help='tell experiment method which task part to execute')
    parser.add_argument('--aid_dir', type=str, default="aid",
                        help="the path for aid data needed in expriments")
    parser.add_argument('--experiment_save_dir', type=str, default="experiment_save_results",
                        help="the path for experiments to save result")

    return parser.parse_args()

configs = set_configs()

dataset = load_data(os.path.join(configs.datapath, configs.dataset))
        
model_configs = {
        'MF': {
                'num_users': int(np.max(dataset["train"].x[:, 0]) + 1),
                'num_items': int(np.max(dataset["train"].x[:, 1]) + 1),
                'embedding_size': 16,
                'weight_decay': 1e-2 # l2 regularization term for training MF or NCF model
            },
        'NCF' : {
                'num_users': int(np.max(dataset["train"].x[:, 0]) + 1),
                'num_items': int(np.max(dataset["train"].x[:, 1]) + 1),
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
        'result_dir': configs.result_dir,
        'model_name': '%s_%s_wd%.0e' % (
            configs.dataset, configs.model, configs.weight_decay)
    }
)

if configs.experiment =="test":
    model.train(verbose=True, plot=True)
    eva_x, eva_y = model.np2tensor(model.dataset['test'].get_batch())
    eva_diff = model.model(eva_x) - eva_y
    print(len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff))
    exit()
elif configs.experiment == "experiment_get_correlation":
    custom_methods.experiment_get_correlation(model=model, configs=configs)
elif configs.experiment == "experiment_remove_all_negtive":
    custom_methods.experiment_remove_all_negtive(model=model, configs=configs)
elif configs.experiment == "experiment_predict_distribution":
    experiment_configs = {
        "precent_to_keep": 0.7
    }
    custom_methods.experiment_predict_distribution(model=model, configs=configs, precent_to_keep=0.7)
elif configs.experiment == "experiment_possible_higher_accuracy":
    experiment_configs = {
        # all task
        "eva_sets": ['test', 'valid'],
        # possible_higher_accuracy_task
        "remain_percents": [0.1, 0.05],
        "repeat_times": 10
    }
    custom_methods.experiment_possible_higher_accuracy(model=model, configs=configs, experiment_configs=experiment_configs)
elif configs.experiment == "experiment_small_model_select_points":
    experiment_configs = {
        # all task
        "eva_sets": ['test', 'valid'],
        # small_model_task
        "remain_percent": 0.1,
        "repeat_times": 10,
        "num_rand_model": 1,
        "data_selecting_method": "mean" # mean or vote
    }
    custom_methods.experiment_small_model_select_points(model=model, configs=configs, experiment_configs=experiment_configs)



