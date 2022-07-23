"""
This python file save method to preproduce some part for 
experiment methods, do the experiment method can be separated 
to different tasks.
"""
from distutils.command.config import config
import os
import shutil
from invoke import task
import numpy as np
from load_data import load_data

def set_configs():
    configs = {
        # detaset
        "dataset_names": ['census_income', 'churn', 'fraud_detection', 'movielens'],
        "datapath": "data",  # the path of datasets
        # Experiment
        "create_tasks": 10, # data processing times
        "epoch_per_task": 10,
        "percents_to_keep": [0.1],
        "save_dir": "aid",
    }
    return configs

def experiment_possible_higher_accuracy(configs):
    save_dir = os.path.join(configs["save_dir"], "experiment_possible_higher_accuracy")
    os.makedirs(save_dir)
    for dataset_name in configs["dataset_names"]:
        dataset = load_data(os.path.join(configs["datapath"], dataset_name))
        task_id = 2
        for percent_to_keep in configs["percents_to_keep"]:
            remain_num = int(dataset['train'].num_examples*percent_to_keep)
            all_select_ids = []
            for i in range(configs["create_tasks"]):
                all_ids_this_task = []
                for epoch in range(configs["epoch_per_task"]):
                    remain_ids = np.random.choice(np.arange(dataset['train'].num_examples), size=remain_num, replace=False)
                    while remain_ids in all_select_ids:
                                remain_ids = np.random.choice(np.arange(dataset['train'].x.shape[0]), size=remain_num, replace=False)
                    all_ids_this_task.append(remain_ids)
                np.savez(
                    os.path.join(save_dir, "{}_task{}".format(dataset_name, task_id)),
                    all_ids_this_task=all_ids_this_task
                )
                task_id += 1
            all_select_ids.append(remain_ids)

def experiment_small_model_select_points(configs):
    save_dir = os.path.join(configs["save_dir"], "experiment_small_model_select_points")
    os.makedirs(save_dir)
    for dataset_name in configs["dataset_names"]:
        dataset = load_data(os.path.join(configs["datapath"], dataset_name))
        task_id = 2
        for percent_to_keep in configs["percents_to_keep"]:
            for i in range(configs["small_model_select_points: num_rand_small_model_for_selecting_data"]):
                remain_num = int(dataset['train'].num_examples*percent_to_keep)
                for i in range(configs["create_tasks"]):
                    all_ids_this_task = []
                    for epoch in range(configs["epoch_per_task"]):
                        remain_ids = np.random.choice(np.arange(dataset['train'].num_examples), size=remain_num, replace=False)
                        all_ids_this_task.append(remain_ids)
            np.savez(
                os.path.join(save_dir, "{}_task{}".format(dataset_name, task_id)),
                all_ids_this_task=all_ids_this_task
            )
            task_id += 1


configs = set_configs()
if os.path.exists(configs["save_dir"]):
    shutil.rmtree(configs["save_dir"])

experiment_small_model_select_points(configs)