"""
This python file save method to preproduce some part for 
experiment methods, do the experiment method can be separated 
to different tasks.
"""
import os
import numpy as np
from load_data import load_data


def experiment_small_model_select_points(configs):
    save_dir = os.path.join(configs["save_dir"], "small_model_select_points")
    if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
    for dataset_name in configs["dataset_names"]:
        dataset = load_data(os.path.join(configs["datapath"], dataset_name))
        remain_num = int(dataset['train'].num_examples*configs["percent_to_keep"])
        all_select_ids = []
        for task_id in range(2, 2+configs["task_num"]):
            task_selected_ids = []
            for epoch in range(configs["epoch_per_task"]):
                remain_ids = np.random.choice(np.arange(dataset['train'].num_examples), size=remain_num, replace=False)
                while remain_ids in all_select_ids:
                            remain_ids = np.random.choice(np.arange(dataset['train'].x.shape[0]), size=remain_num, replace=False)
                task_selected_ids.append(remain_ids)
            np.savez(
                os.path.join(save_dir, "{}_task{}".format(dataset_name, task_id)),
                remain_ids=remain_ids
            )
        all_select_ids.append(remain_ids)


configs = {
        # detaset
        "dataset_names": ['census_income', 'churn', 'fraud_detection', 'movielens'],
        "datapath": "data",  # the path of datasets
        "percent_to_keep": 0.1,
        "task_num": 10, # data processing times
        "epoch_per_task": 10,
        "save_dir": "aid"
    }
experiment_small_model_select_points(configs)