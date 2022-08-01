'''
This python file is used to save special methods in the research

'''

import torch
import numpy as np
import os
from scipy.stats import pearsonr
from math import ceil
import time, threading

# method
def chack_eva_type():
    pass

def hessian(model):
        train_x, train_y = model.np2tensor(model.dataset['train'].get_batch())
        train_loss = model.loss_fn(model.model(train_x), train_y)
        fir_grads = torch.autograd.grad(train_loss, model.model.parameters(), create_graph=True)
        fir_grads = torch.cat([grad.flatten() for grad in fir_grads])

        sec_grads = []
        for grad in fir_grads:
            partial_grad = torch.autograd.grad(grad, model.model.parameters(), create_graph=True)
            partial_grad = torch.cat([grad.flatten() for grad in partial_grad])
            sec_grads.append(partial_grad)
        sec_grads = torch.stack(sec_grads)
        return sec_grads

def inverse_matrix(matrix):
    if np.linalg.det(matrix) == 0:
        return np.linalg.pinv(matrix)
    inverse_matrix = np.identity(len(matrix))
    for I_col in range(len(matrix)):
        inverse_matrix[I_col, :] /= matrix[I_col, I_col]
        matrix[I_col, :] /= matrix[I_col, I_col]
        for row in range(len(matrix)):
            if I_col != row:
                inverse_matrix[row, :] -= inverse_matrix[I_col, :]* matrix[row, I_col]
                matrix[row, :] -= matrix[I_col, :]* matrix[row, I_col]
    return inverse_matrix

def predict_on_single(model=None, eva_id=None, eva_set_type='test', inf_id=None):
        '''This method predict difference will appear on the evaluate single point performance(ex: loss, y)
        after add one single point in train set. (add_one_point_model - origin_model)

        Args:
            model (_type_, optional): _description_. Defaults to None.
            eva_id (_type_, optional): _description_. Defaults to None.
            eva_set_type (_type_, optional): _description_. Defaults to 'test'.
            inf_id (_type_, optional): _description_. Defaults to None.
        '''
        assert model is not None
        assert eva_id is not None
        assert eva_set_type in ['train', 'test', 'valid']
        assert inf_id is not None
        eva_x, eva_y = model.np2tensor(model.dataset[eva_set_type].get_by_idxs(eva_id))
        eva_loss = model.loss_fn(model.model(eva_x), eva_y)
        eva_grad = torch.autograd.grad(eva_loss, model.model.parameters())
        eva_grad = torch.cat([grad.flatten() for grad in eva_grad]).numpy()
        inverse_H = inverse_matrix(hessian(model).detach().numpy())

        inf_x, inf_y = model.np2tensor(model.dataset['train'].get_by_idxs(inf_id))
        inf_loss = model.loss_fn(model.model(inf_x), inf_y)
        inf_grad = torch.autograd.grad(inf_loss, model.model.parameters())
        inf_grad = torch.cat([grad.flatten() for grad in inf_grad]).numpy()

        inf = -np.matmul(eva_grad, np.matmul(inverse_H, inf_grad))/ model.dataset['train'].num_examples
        return inf

def predict_on_batch(model=None, eva_set_type='test', inf_id=None):
    '''This method predict difference will appear on the evaluate dataset performance(ex: loss)
        after add one single point in train set. (add_one_point_model - origin_model)

    Args:
        verbose (bool, optional): _description_. Defaults to True.
        target_fn (_type_, optional): _description_. Defaults to None.
        test_id (_type_, optional): _description_. Defaults to None.
        removed_id (_type_, optional): _description_. Defaults to None.
    '''
    assert model is not None
    assert eva_set_type in ['train', 'test', 'valid']
    assert inf_id is not None

    eva_x, eva_y = model.np2tensor(model.dataset[eva_set_type].get_batch())
    eva_loss = model.loss_fn(model.model(eva_x), eva_y)
    eva_grad = torch.autograd.grad(eva_loss, model.model.parameters())
    eva_grad = torch.cat([grad.flatten() for grad in eva_grad]).numpy()
    inverse_H = inverse_matrix(hessian(model).detach().numpy())

    inf_x, inf_y = model.np2tensor(model.dataset['train'].get_by_idxs(inf_id))
    inf_loss = model.loss_fn(model.model(inf_x), inf_y)
    inf_grad = torch.autograd.grad(inf_loss, model.model.parameters())
    inf_grad = torch.cat([grad.flatten() for grad in inf_grad]).numpy()

    inf = -np.matmul(eva_grad, np.matmul(inverse_H, inf_grad)) / model.dataset['train'].num_examples
    return inf

# experience
def experiment_get_correlation(model, configs, eva_set_type='test', eva_id=None):
    '''This method gets the correlation between predict influences and real influences
    of all current sample points.

    Args:
        model (Model): An model object
        configs (argparse.Namespace): An argparse.Namespace object includes all configurations.
        eva_set_type(list): An List of strings represents all types of datasets to be evaluated.
        eva_id(int): An integer represents the id of data point to be evaluated.
    '''
    assert eva_set_type in ['train', 'test', 'valid']
    pred_diffs = []
    real_diffs = []

    # get original model
    checkpoint = model.train(
        num_epoch=configs.num_epoch_train,
        verbose=True,
        checkpoint_name='Test'
    )
    if eva_id is None:
        eva_x, eva_y = model.np2tensor(model.dataset[eva_set_type].get_batch())
    else:
        eva_x, eva_y = model.np2tensor(model.dataset[eva_set_type].get_by_idxs(eva_id))
    ori_loss = model.loss_fn(model.model(eva_x), eva_y).item()

    for inf_id in range(model.dataset['train'].num_examples):
        print('processing point {}/{}'.format(inf_id+1, model.dataset['train'].num_examples))
        if eva_id is None:
            pred_diffs.append(predict_on_batch(
                model=model,
                eva_set_type=eva_set_type,
                inf_id=inf_id
            ))
        else:
            pred_diffs.append(predict_on_single(
                model=model,
                eva_set_type=eva_set_type,
                eva_id=eva_id,
                inf_id=inf_id  
            ))
        remain_ids = np.append(model.remain_ids, np.array([inf_id]))
        # remain_ids = np.setdiff1d(model.remain_ids, np.array([inf_id]))
        model.reset_train_dataset(remain_ids)
        model.train(
            num_epoch=configs.num_epoch_train,
            checkpoint_name='eva{}_+inf{}'.format(eva_id, inf_id)
        )
        re_loss = model.loss_fn(model.model(eva_x), eva_y).item()
        real_diffs.append(re_loss - ori_loss)
        model.load_model(checkpoint)
        
    real_diffs = np.array(real_diffs)
    pred_diffs = np.array(pred_diffs)
    print('Correlation is %s' % pearsonr(real_diffs, pred_diffs)[0])
    if os.path.exists(configs.experiment_save_dir) is False:
            os.makedirs(configs.experiment_save_dir)
    np.savez(
        '{}/result-{}-{}.npz'.format(configs.experiment_save_dir, configs.model, configs.dataset),
        real_diffs=real_diffs,
        pred_diffs=pred_diffs,
    )

def experiment_predict_distribution(model, configs, precent_to_keep=1.0, epoch=100, eva_set_type='test'):
    all_select_ids = []
    samples_diff_dic = {}
    remain_num = int(model.dataset['train'].x.shape[0]*precent_to_keep)
    for i in range(epoch):
        remain_ids = np.random.choice(np.arange(model.dataset['train'].x.shape[0]), size=remain_num, replace=False)
        while remain_ids in all_select_ids:
            remain_ids = np.random.choice(np.arange(model.dataset['train'].x.shape[0]), size=remain_num, replace=False)
        all_select_ids.append(remain_ids)
        model.reset_train_dataset(remain_ids)
        model.train(
            num_epoch=configs.num_epoch_train,
            checkpoint_name='rand{}_num{}'.format(i, remain_num)
        )
        for inf_id in range(model.dataset['train'].x.shape[0]):
            if inf_id not in samples_diff_dic.keys():
                samples_diff_dic[inf_id] = [predict_on_batch(model, eva_set_type, inf_id)]
            else:
                samples_diff_dic[inf_id].append(predict_on_batch(model, eva_set_type, inf_id))
    if os.path.exists(configs.experiment_save_dir) is False:
            os.makedirs(configs.experiment_save_dir)
    np.savez(
        '{}/rand{}-{}-{}.npz'.format(configs.experiment_save_dir, precent_to_keep, configs.model, configs.dataset),
        point_diffs=samples_diff_dic
    )

def experiment_possible_higher_accuracy(model, configs, experiment_configs):
    '''
    This experiment evaluates the performance of small models with certain percent of randomly selected data. The task step is the following:
        1. get accuracies of a big model(with all training data) on all eva_sets.
        2. train small models with {remain_percent} percent randomly selected data with {num_small_model_performance_task}.
            remain_percent comes from remain_percents

    Args:
        model (Model): An model object
        configs (argparse.Namespace): An argparse.Namespace object includes all configurations.
        experiment_configs(dict): An dictionary of experiment_configs.
    '''
    
    def whole_data_accuracies_task():
        '''
        This task get accuracies of a big model(with all training data) on all eva_sets.
        
        this task can be tested by the following code:
            python -u main.py --experiment experiment_possible_higher_accuracy --dataset fraud_detection --task_num 1
        '''
        
        performance = {}
        model.train(num_epoch=configs.num_epoch_train, checkpoint_name='ori', verbose=True)
        for eva_type in experiment_configs['eva_sets']:
            eva_x, eva_y = model.np2tensor(model.dataset[eva_type].get_batch())
            eva_diff = model.model(eva_x) - eva_y
            if eva_type in performance.keys():
                performance[eva_type].append(len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff))
            else:
                performance[eva_type] = [len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff)]

        if os.path.exists(configs.experiment_save_dir) is False:
            os.makedirs(configs.experiment_save_dir)
        np.savez(
            '{}/{}_task{}.npz'.format(configs.experiment_save_dir, configs.dataset, configs.task_num),
            performance=performance
        )

    def small_model_performance_task():
        '''
        This task gets accuracies of a small model(with a certain percent data randomly selected from all data) on all eva_sets.
        
        this task can be tested by the following code:
            python -u main.py --experiment experiment_possible_higher_accuracy --dataset fraud_detection --task_num 2
        '''
        
        performance = {}
        remain_num = int(model.dataset['train'].x.shape[0]*experiment_configs['remain_percent'])
        remain_ids = np.random.choice(np.arange(model.dataset['train'].x.shape[0]), size=remain_num, replace=False)
        model.reset_train_dataset(remain_ids)
        model.train(
            num_epoch=configs.num_epoch_train,
            load_checkpoint=configs.load_checkpoint,
            save_checkpoint=configs.save_checkpoint,
        )
        for eva_type in experiment_configs['eva_sets']:
            eva_x, eva_y = model.np2tensor(model.dataset[eva_type].get_batch())
            eva_diff = model.model(eva_x) - eva_y
            if eva_type in performance.keys():
                performance[eva_type].append(len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff))
            else:
                performance[eva_type] = [len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff)]

        if os.path.exists(configs.experiment_save_dir) is False:
            os.makedirs(configs.experiment_save_dir)
        np.savez(
            '{}/{}_task{}.npz'.format(configs.experiment_save_dir, configs.dataset, configs.task_num),
            performance=performance
        )

    for eva_set_type in experiment_configs['eva_sets']:
        assert eva_set_type in ['train', 'test', 'valid']

    part1_end_num = experiment_configs['num_whole_data_accuracies_task']
    part2_end_num = part1_end_num + experiment_configs['num_small_model_performance_task']*len(experiment_configs['remain_percents'])

    if configs.task_num > 0 and configs.task_num <= part1_end_num:
        whole_data_accuracies_task()

    elif configs.task_num <= part2_end_num:
        remain_percent_id = int((configs.task_num - part1_end_num) / experiment_configs['num_small_model_performance_task'])
        experiment_configs['remain_percent'] = experiment_configs['remain_percents'][remain_percent_id]
        small_model_performance_task()

    else:
        assert configs.task_num <= part2_end_num \
            , 'Task number {} is more than maximum {}'.format(configs.task_num, part2_end_num)

def experiment_small_model_select_points(model, configs, experiment_configs):
    '''
    This experiment evaluates the performance of small model with well selected data comparing with the model with the whole model. The task step is following:
        1. get accuracies of a big model(with all training data) on all eva_sets.
        2. train small models with randomly selected data.
        3. get indexes of selected data according to the influence value predicted by previous models.
            
        4. the influence value of each data point will be sorted
            A certain percent of data will be remained with the most negative influence value(adding the point has the most negative effect on loss)
            two method will be applied for selecting data points: mean value method, vote method
                mean method: count the mean influence value as the predicted influence value
                vote method: count rejection rate as the predicted influence value
            train small models with well selected data and get the accuracy of the model on all eva_sets
    
    this task can be tested by the following code:
        For part 1
            python -u main.py --experiment experiment_small_model_select_points --dataset churn --task_num 1
        For part 2
            python -u main.py --experiment experiment_small_model_select_points --dataset churn --task_num 2
        For part 3 (Need to finish previous parts)
            #### step 1 1 ~ 5000 for 6000 data points
            python -u main.py --experiment experiment_small_model_select_points --dataset churn --task_num 102
            #### step 2 5000 ~ 6000 for 6000 data points
            python -u main.py --experiment experiment_small_model_select_points --dataset churn --task_num 202
        For part 4 (Need to finish previous parts, and experiment_configs['num_conbined_inf'] == 1)
            python -u main.py --experiment experiment_small_model_select_points --dataset churn --task_num 302

    Args:
        model (Model): An model object
        configs (argparse.Namespace): An argparse.Namespace object includes all configurations.
        experiment_configs(dict): An dictionary of experiment_configs.
    '''
    def whole_data_accuracies_task():
        '''
        This task gets accuracies of a big model(with all training data) on all eva_sets.
        '''
        
        print(1, part1_end_num+1, part2_end_num+1, part3_end_num+1, '{}(End)'.format(part4_end_num+1))

        performance = {}
        model.train(
            num_epoch=configs.num_epoch_train,
            checkpoint_name='big{}_num{}'.format((configs.task_num), model.dataset['train'].num_examples),
        )
        for eva_type in experiment_configs['performance_eva_sets']:
            eva_x, eva_y = model.np2tensor(model.dataset[eva_type].get_batch())
            eva_diff = model.model(eva_x) - eva_y
            performance[eva_type] = [len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff)]
        print('performance: ', str(performance))
        
    
    def rand_model_task():
        '''
        This task trains a small model with randomly selected data.
        '''
        remain_num = int(model.dataset['train'].x.shape[0]*experiment_configs['rand_remain_percent'])
        remain_ids = np.random.choice(np.arange(model.dataset['train'].x.shape[0]), size=remain_num, replace=False)

        model.reset_train_dataset(remain_ids)
        model.train(
            num_epoch=configs.num_epoch_train,
            checkpoint_name='rand{}_num{}'.format((configs.task_num-part1_end_num), remain_num),
            load_checkpoint=False,
            save_checkpoint=True,
            plot=False
        )

    def rand_model_inf_task():
        '''
        This task calculates influence value of {max_data_points_num_to_calculate} data points based on previous random-data model.
        '''
        current_step = ceil((configs.task_num - part2_end_num)/experiment_configs['num_rand_model_task'])
        total_step = ceil(model.dataset['train'].x.shape[0]/experiment_configs['max_data_points_num_to_calculate'])
        print('step {}/{}'.format(current_step, total_step))
        
        rand_id = (configs.task_num - part2_end_num - 1)%experiment_configs['num_rand_model_task'] + 1
        remain_num = int(model.dataset['train'].x.shape[0]*experiment_configs['rand_remain_percent'])
        print('load model: rand{}_num{}___{}_step{}'.format(rand_id, remain_num, model.model_name, configs.num_epoch_train))
        model.load_model('rand{}_num{}___{}_step{}'.format(rand_id, remain_num, model.model_name, configs.num_epoch_train))
        
        infs = np.array([])
        save_dir = '{}/{}_rand{}_infs'.format(configs.experiment_save_dir, configs.dataset, rand_id)
        infs_file = '{}/{}-{}.npz'.format(save_dir, current_step, total_step)
        
        start_data_id = (current_step-1)*experiment_configs['max_data_points_num_to_calculate']
        end_data_id = min(current_step*experiment_configs['max_data_points_num_to_calculate'], model.dataset['train'].x.shape[0])
        for inf_id in range(start_data_id, end_data_id):
            print('processing point {}/{}'.format(inf_id+1, model.dataset['train'].x.shape[0]))
            infs = np.append(infs, -predict_on_batch(
                    model=model,
                    eva_set_type=experiment_configs['inf_eva_set'],
                    inf_id=inf_id
                ))
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        np.savez(
            infs_file,
            infs=infs
        )
    
    def selected_model_task():
        '''
        This task trains a selected-data small model with a {select_remain_percent} percent data points and evaluates its performance.
        Data are selected based on {num_combined_inf} small rand models.
        '''
        all_infs = []

        selected_id = configs.task_num - part3_end_num
        for i in range(experiment_configs['num_combined_inf']-1, -1, -1):
            rand_id = selected_id*experiment_configs['num_combined_inf'] - i
            total_step = ceil(model.dataset['train'].x.shape[0]/experiment_configs['max_data_points_num_to_calculate'])
            infs = np.array([])
            for i in range(1, total_step+1):
                infs_file = '{}/{}_rand{}_infs/{}-{}.npz'.format(configs.experiment_save_dir, configs.dataset, rand_id, i, total_step)
                if not os.path.exists(infs_file):
                    print('Waiting: {}'.format(infs_file))
                while not os.path.exists(infs_file):
                    continue
                infs = np.append(infs, np.load(infs_file)['infs'])
                print('File: {}_rand{}_infs/{}-{}.npz'.format(configs.dataset, rand_id, i, total_step))
            print(' - loaded data points [{}]'.format(len(infs)))
            all_infs.append(infs)

        performance = {}
        all_infs = np.stack(all_infs)
        remain_num = int(model.dataset['train'].x.shape[0]*experiment_configs['rand_remain_percent'])
        if experiment_configs['data_selecting_method'] == 'mean':
            inf_sorted_ids = np.argsort(np.mean(all_infs, axis=0))
            top_inf_ids = inf_sorted_ids[:remain_num]
            
        elif experiment_configs['data_selecting_method'] == 'vote':
            inf_sorted_ids = np.argsort(np.count_nonzero(all_infs > 0, axis=0)/experiment_configs['num_combined_inf'])
            top_inf_ids = inf_sorted_ids[:remain_num]
        else:
            assert NotImplementedError
        
        model.reset_train_dataset(top_inf_ids)
        model.train(
            num_epoch=configs.num_epoch_train,
            checkpoint_name='select{}_num{}'.format(selected_id, top_inf_ids),
            save_checkpoint=True        
        )

        # evaluate the accuracies of the model on all eva_set
        for eva_type in experiment_configs['performance_eva_sets']:
            eva_x, eva_y = model.np2tensor(model.dataset[eva_type].get_batch())
            eva_diff = model.model(eva_x) - eva_y
            if eva_type in performance.keys():
                performance[eva_type].append(len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff))
            else:
                performance[eva_type] = [len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff)]
        # save the result
        if os.path.exists(configs.experiment_save_dir) is False:
            os.makedirs(configs.experiment_save_dir)
        np.savez(
            '{}/{}_select{}_performance.npz'.format(configs.experiment_save_dir, configs.dataset, selected_id),
            performance=performance
        )
        print(performance)

    for eva_set_type in experiment_configs['performance_eva_sets']:
        assert eva_set_type in ['train', 'test', 'valid']
    
    part1_end_num = experiment_configs['num_whole_data_accuracies_task']
    part2_end_num = part1_end_num + experiment_configs['num_rand_model_task']
    part3_end_num = part2_end_num + experiment_configs['num_rand_model_task']*ceil(model.dataset['train'].x.shape[0]/experiment_configs['max_data_points_num_to_calculate'])
    part4_end_num = part3_end_num + ceil(experiment_configs['num_rand_model_task'] / experiment_configs['num_combined_inf'])

    if configs.task_num > 0 and configs.task_num <= part1_end_num:
        whole_data_accuracies_task()
    elif configs.task_num <= part2_end_num:
        rand_model_task()
    elif configs.task_num <= part3_end_num:
        rand_model_inf_task()
    elif configs.task_num <= part4_end_num:
        selected_model_task()
    else:
        assert configs.task_num <= part4_end_num \
            , 'Task number {} is more than maximum {}'.format(configs.task_num, part4_end_num)

def experiment_small_model_based_big_model(model, configs, experiment_configs):
    """
    This experiment evaluates the performance of small model based on influence value predicted by big model which has all data.
        1. train a big model
        2. get influence value of all data based on the big model
        3. get performance of a {} percent small model

    Args:
        model (Model): An model object
        configs (argparse.Namespace): An argparse.Namespace object includes all configurations.
        experiment_configs(dict): An dictionary of experiment_configs.
    """

    def whole_data_accuracies_task():
        '''
        This task gets accuracies of a big model(with all training data) on all eva_sets.
        '''
        
        print(1, part1_end_num+1, part2_end_num, '{}(End)'.format(part3_end_num))

        model.train(
            num_epoch=configs.num_epoch_train,
            checkpoint_name='big_num{}'.format(model.dataset['train'].num_examples),
            save_checkpoint=True
        )

        performance = {}
        # evaluate the accuracies of the model on all eva_set
        for eva_type in experiment_configs['performance_eva_sets']:
            eva_x, eva_y = model.np2tensor(model.dataset[eva_type].get_batch())
            eva_diff = model.model(eva_x) - eva_y
            if eva_type in performance.keys():
                performance[eva_type].append(len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff))
            else:
                performance[eva_type] = [len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff)]
        print(performance)

    
    def all_inf_task():
        '''
        This task calculates influence value of {max_data_points_num_to_calculate} data points based on previous big model.
        '''
        current_step = configs.task_num - part1_end_num
        total_step = ceil(model.dataset['train'].x.shape[0]/experiment_configs['max_data_points_num_to_calculate'])
        print('step {}/{}'.format(current_step, total_step))
        
        print('load model: big_num{}___{}_step{}'.format(model.dataset['train'].num_examples, model.model_name, configs.num_epoch_train))
        model.load_model('big_num{}___{}_step{}'.format(model.dataset['train'].num_examples, model.model_name, configs.num_epoch_train))
        
        infs = np.array([])
        save_dir = '{}/{}_infs'.format(configs.experiment_save_dir, configs.dataset)
        infs_file = '{}/{}-{}.npz'.format(save_dir, current_step, total_step)
        
        start_data_id = (current_step-1)*experiment_configs['max_data_points_num_to_calculate']
        end_data_id = min(current_step*experiment_configs['max_data_points_num_to_calculate'], model.dataset['train'].x.shape[0])
        for inf_id in range(start_data_id, end_data_id):
            print('processing point {}/{}'.format(inf_id+1, model.dataset['train'].x.shape[0]))
            infs = np.append(infs, -predict_on_batch(
                    model=model,
                    eva_set_type=experiment_configs['inf_eva_set'],
                    inf_id=inf_id
                ))
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        np.savez(
            infs_file,
            infs=infs
        )

    def small_model_task():
        """
        This task trains a selected-data small model with a {select_remain_percent} percent data points and evaluates its performance.
        Data are selected based on the previous big models.
        """

        total_step = ceil(model.dataset['train'].x.shape[0]/experiment_configs['max_data_points_num_to_calculate'])
        infs = np.array([])
        for i in range(1, total_step+1):
            infs_file = '{}/{}_infs/{}-{}.npz'.format(configs.experiment_save_dir, configs.dataset, i, total_step)
            if not os.path.exists(infs_file):
                print('Waiting: {}'.format(infs_file))
            while not os.path.exists(infs_file):
                continue
            infs = np.append(infs, np.load(infs_file)['infs'])
            print('File: {}_infs/{}-{}.npz'.format(configs.dataset, i, total_step))
        print(' - loaded data points [{}]'.format(len(infs)))

        performance = {}
        remain_num = int(model.dataset['train'].x.shape[0]*experiment_configs['remain_percent'])
        
        inf_sorted_ids = np.argsort(infs)
        top_inf_ids = inf_sorted_ids[:remain_num]
        
        model.reset_train_dataset(top_inf_ids)
        model.train(
            num_epoch=configs.num_epoch_train,
            checkpoint_name='select_num{}'.format(model.dataset['train'].num_examples),
        )

        # evaluate the accuracies of the model on all eva_set
        for eva_type in experiment_configs['performance_eva_sets']:
            eva_x, eva_y = model.np2tensor(model.dataset[eva_type].get_batch())
            eva_diff = model.model(eva_x) - eva_y
            if eva_type in performance.keys():
                performance[eva_type].append(len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff))
            else:
                performance[eva_type] = [len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff)]
        print(performance)


    for eva_set_type in experiment_configs['performance_eva_sets']:
        assert eva_set_type in ['train', 'test', 'valid']
    
    part1_end_num = 1
    part2_end_num = part1_end_num + ceil(model.dataset['train'].x.shape[0]/experiment_configs['max_data_points_num_to_calculate'])
    part3_end_num = part2_end_num + 1

    if configs.task_num > 0 and configs.task_num <= part1_end_num:
        whole_data_accuracies_task()
    elif configs.task_num <= part2_end_num:
        all_inf_task()
    elif configs.task_num <= part3_end_num:
        small_model_task()
    else:
        assert configs.task_num <= part2_end_num \
            , 'Task number {} is more than maximum {}'.format(configs.task_num, part2_end_num)

def experiment_remove_all_negtive(model, configs, eva_set_type='test', eva_id=None, based_pred=True):
    assert eva_set_type in ['train', 'test', 'valid']

    diffs = np.array([-1])
    samples_diff_dic = {}
    while len(diffs<=0) > 0:
        # get original checkpoint
        model.train(
                num_epoch=configs.num_epoch_train,
                load_checkpoint=configs.load_checkpoint,
                save_checkpoint=configs.save_checkpoint,
                checkpoint_name='ori_num{}'.format(model.dataset['train'].num_examples)
        )
        if not based_pred:
            # get test data
            if eva_id is None:
                test_x, test_y = model.np2tensor(model.dataset['test'].get_batch())
            else:
                test_x, test_y = model.np2tensor(model.dataset['test'].get_by_idxs(eva_id))
            ori_loss = model.loss_fn(model.model(test_x), test_y).item()
        
        diffs = np.array([])
        for inf_id in range(model.dataset['train'].x.shape[0]):
            print('Remain {} data points'.format(model.dataset['train'].num_examples))
            print('processing point {}/{}'.format(inf_id+1, model.dataset['train'].x.shape[0]))
            if based_pred:
                if eva_id is None:
                    diffs = np.append(diffs, -predict_on_batch(
                        model=model,
                        eva_set_type=eva_set_type,
                        inf_id=inf_id
                    ))
                else:
                    diffs = np.append(diffs, -predict_on_single(
                        model=model,
                        eva_id=eva_id,
                        eva_set_type=eva_set_type,
                        inf_id=inf_id
                    ))
            else:
                if inf_id in remain_ids:
                    remain_ids = np.setdiff1d(model.remain_ids, np.array([inf_id]))
                    model.reset_train_dataset(remain_ids)
                    model.train(
                        num_epoch=configs.num_epoch_train,
                        load_checkpoint=configs.load_checkpoint,
                        save_checkpoint=configs.save_checkpoint,
                        verbose=False,
                        checkpoint_name='eva{}_inf{}_num{}'.format(eva_id, inf_id, model.dataset['train'].num_examples),
                        plot=configs.plot
                    )
                    re_loss = model.loss_fn(model.model(test_x), test_y).item()
                    diffs = np.append(diffs, (re_loss - ori_loss))
                else:
                    diffs = np.append(diffs, np.nan)
            if inf_id in samples_diff_dic.keys():
                samples_diff_dic[inf_id].append(diffs[inf_id])
            else:
                samples_diff_dic[inf_id] = [diffs[inf_id]]
        copy_diffs = diffs[model.remain_ids]
        model.remain_ids = model.remain_ids[np.argsort(copy_diffs)]
        print('remove point {}'.format(model.remain_ids[0]))
        model.reset_train_dataset(model.remain_ids[1:])
        diffs = diffs[model.remain_ids]
    if os.path.exists(configs.experiment_save_dir) is False:
            os.makedirs(configs.experiment_save_dir)
    np.savez(
        '{}/remove_all_negtive-{}-{}.npz'.format(configs.experiment_save_dir, configs.model, configs.dataset),
        ids=model.remain_ids,
        samples_diff_dic=samples_diff_dic
    )
