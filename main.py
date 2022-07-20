import numpy as np
import os
import torch
import argparse
from load_data import load_data
from model.generic_neural_net import Model
import custom_method

def parse_args():
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
    # train process
    parser.add_argument('--num_epoch_train', type=int, default=270000,
                        help='training steps for the training model')
    parser.add_argument('--load_checkpoint', type=bool, default=True,
                        help='whether loading previous model if it exists')
    parser.add_argument('--save_checkpoint', type=bool, default=True,
                        help='whether saving the current model')
    parser.add_argument('--plot', type=bool, default=False,
                        help='if plot the figure of train loss and test loss')
    # Influence on single point by remove one data point
    parser.add_argument('--single_point', nargs='+', type=str, default=["test", "test_y"],
                        help='the target y to be evaluated, train_y, train_loss, test_y, test_loss, None. None means not to evaluate.')
    # experiment
    parser.add_argument('--task_num', type=int, default=1,
                        help='tell experiment method which part to execute')

    return parser.parse_args()


configs = parse_args()

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
model.train(verbose=True, plot=True)
eva_x, eva_y = model.np2tensor(model.dataset['test'].get_batch())
eva_diff = model.model(eva_x) - eva_y
print(len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff))
exit()

# custom_method.experiment_get_correlation(model=model, configs=configs)
# custom_method.experiment_remove_all_negtive(model=model, configs=configs)
# custom_method.experiment_predict_distribution(model, configs, precent_to_keep=0.7)
# custom_method.experiment_possible_better(model, configs, percents=[0.1, 0.05,], eva_set=['test', 'valid'])
custom_method.experiment_small_model_select_points()



