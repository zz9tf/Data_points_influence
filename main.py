import numpy as np
import torch
import os
from load_data import load_data
from model.generic_neural_net import Model
import custom_method

configs = {
    # detaset
    "dataset": "ratio",  # name of dataset: movielens or yelp or census_income
    "datapath": "data",  # the path of datasets
    # model configs
    "model": "lr",  # modeltype:MF or NCF or lr
    "embedding_size": 2,  # embedding size
    # train configs
    "batch_size": 4096,  # 3020,  # the batch_size for training or predict, None for not to use batch
    "lr": 1e-7,  # initial learning rate for training MF or NCF model
    "weight_decay": 1e-2,  # l2 regularization term for training MF or NCF model
    # train
    "num_epoch_train": 270000,  # training steps
    "load_checkpoint": True,  # whether loading previous model if it exists.
    "save_checkpoint": True,  # whether saving the current model
    "plot": False ,  # if plot the figure of train loss and test loss
    # Influence on single point by remove one data point
    "single_point": ["test", "test_y"],   # the target y to be evaluated, train_y, train_loss, test_y, test_loss, None. None means not to evaluate.
    "num_of_single": 50,  # the number of data points to be removed to evaluate the influence
    # Influence on loss by remove one data point
    "batch_points": None,  # the target loss function to be evaluated, train, test, None. None means not to evaluate.
    
    "num_to_removed": 5,  # number of points to retrain
    "retrain_times": 4,  # times to retrain the model
    "percentage_to_keep": [1],  # A list of the percentage of training dataset to keep, ex: 0.3, 0.5, 0.7, 0.9
}


dataset = load_data(os.path.join(configs['datapath'], configs['dataset']))

num_users = int(np.max(dataset["train"].x[:, 0]) + 1)
num_items = int(np.max(dataset["train"].x[:, 1]) + 1)

model = Model(
    # model
    model_configs={
        'num_users': num_users,
        'num_items': num_items,
        'embedding_size': configs['embedding_size'],
        'weight_decay': configs['weight_decay'],
    },
    basic_configs={
        'model': configs['model'],
        # loading data
        'dataset': dataset,
        # train configs
        'batch_size': configs['batch_size'],
        'learning_rate': configs['lr'],
        'weight_decay': configs['weight_decay'],
        
        # loading configs
        'result_dir': 'result',
        'model_name': '%s_%s_embed%d_wd%.0e' % (
            configs['dataset'], configs['model'], configs['embedding_size'], configs['weight_decay'])
    }
)
model.train(verbose=True, plot=True)
eva_x, eva_y = model.np2tensor(model.dataset['test'].get_batch())
eva_diff = model.model(eva_x) - eva_y
print(eva_diff)
print(eva_y)
print(torch.mean(1-torch.abs(eva_diff/eva_y)))
exit()

# custom_method.experience_get_correlation(model=model, configs=configs)
# custom_method.exprience_remove_all_negtive(model=model, configs=configs)
# custom_method.experience_predict_distribution(model, configs, precent_to_keep=0.7)
custom_method.experience_possible_better(model, configs, percents=[0.9, 0.7, 0.5, 0.3, 0.1], eva_set=['test', 'valid']) # 


