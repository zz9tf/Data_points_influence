import sys
sys.path.append('../../pytorchGenericNet')
import numpy as np
import pandas as pd
from  untils.load_data import load_data
from model.generic_neural_net import Model
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from math import ceil

# method
def set_configs():
    # configs perparing...
    configs = {
        # detaset
        'dataset': 'churn',  # name of dataset: movielens, yelp, census_income, churn, fraud_detection
        'datapath': '../data',  # the path of datasets
        # model configs
        'model': 'lr',  # model type: MF or NCF or lr
        # train configs
        'batch_size': 2048,  # 3020,  # the batch_size for training or predict, None for not to use batch
        'lr': 1e-4,  # initial learning rate for training MF or NCF model
        'weight_decay': 1e-2,  # l2 regularization term for training MF or NCF model
        # train
        'num_epoch_train': 30000,  # training steps
        'load_checkpoint': True,  # whether loading previous model if it exists.
        'save_checkpoint': True,  # whether saving the current model
        'plot': False ,  # if plot the figure of train loss and test loss
        # Influence on single point by remove one data point
        'single_point': ['test', 'test_y'],   # the target y to be evaluated, train_y, train_loss, test_y, test_loss, None. None means not to evaluate.
    }

    dataset = load_data(os.path.join(configs['datapath'], configs['dataset']))
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
        model_configs=model_configs[configs['model']],
        basic_configs={
            'model': configs['model'],
            # loading data
            'dataset': dataset,
            # train configs
            'batch_size': configs['batch_size'],
            'learning_rate': configs['lr'],
            'weight_decay': configs['weight_decay'],
            
            # loading configs
            'checkpoint_dir': 'd:\experiment_small_model_select_points\checkpoints',
            'model_name': '%s_%s_embed%d_wd%.0e' % (
                configs['dataset'], configs['model'], 2, configs['weight_decay'])
        }
    )

    read_configs = {
        'read_dir': '../experiment_results_storage',
        'configs': configs,
        'model': model,
        'model_configs': model_configs,
        'performance_eva_sets': ['test', 'valid']
    }
    return read_configs

def violin_plot(main_color, line_color, scatter_color
            , all_data, x_axis_labels, ax
            , title=None, x_label=None, y_label=None):
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    def set_axis_style(ax, labels):
        ax.xaxis.set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
    
    try:
        all_data = [np.sort(data) for data in all_data]
    except:
        all_data = [np.sort(all_data)]

    parts = ax.violinplot(all_data, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(main_color)
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    quartile1, medians, quartile3 = [], [], []
    for data in all_data:
        q1, m, q3 = np.percentile(data, [25, 50, 75], axis=0)
        quartile1.append(q1)
        medians.append(m)
        quartile3.append(q3)
    whiskers = np.array([
        adjacent_values(data, q1, q3)
        for data, q1, q3 in zip(all_data, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color=scatter_color, s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color=line_color, linestyle='-', lw=5)
    ax.vlines(inds, whiskers_min, whiskers_max, color=line_color, linestyle='-', lw=2)
    set_axis_style(ax, x_axis_labels)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)

# read
def read_experiment_get_correlation(configs):
    diff_dic = np.load('{}/correlation/result-lr-movielens-remove.npz'.format(configs['read_dir']))

    plt.scatter(diff_dic['real_diffs'], diff_dic['pred_diffs'])
    all_min = min(min(diff_dic['real_diffs']), min(diff_dic['pred_diffs']))
    all_max = max(max(diff_dic['real_diffs']), max(diff_dic['pred_diffs']))
    plt.plot([all_min, all_max], [all_min, all_max], color='blue', label='x=y')
    real_min = np.min(diff_dic['real_diffs'])
    real_max = np.max(diff_dic['real_diffs'])
    predict_min = np.min(diff_dic['pred_diffs'])
    predict_max = np.max(diff_dic['pred_diffs'])
    plt.plot([real_min, real_max], [predict_min, predict_max], color='red', label='min2max on two D')
    plt.plot([real_min, real_max], [0, 0], color='orange', label='predict effect dividing line')
    plt.plot([0, 0], [predict_min, predict_max], color='orange', label='real effect dividing line')
    plt.xlabel('real diff')
    plt.ylabel('predict diff')
    plt.legend()
    plt.show()

def read_remove_all_negtive(read_configs):
    remove_all_dic = np.load('{}/remove_all_negtive-lr-movielens.npz'.format(read_configs['read_dir']), allow_pickle=True)
    print(remove_all_dic['ids'])
    samples_diff_dic = remove_all_dic['samples_diff_dic'].item()
    for item in samples_diff_dic.items():
        plt.plot(np.arange(len(item[1])), item[1], label=str(item[0]))
        plt.xlabel('num of remove points')
        plt.ylabel('diff')
        plt.legend()
        plt.show()

def read_inf_variance_change_with_randTime(read_configs):
    point_diffs = np.load('{}/inf_variance_with_remains/rand0.1-lr-movielens.npz'.format(read_configs['read_dir']), allow_pickle=True)['point_diffs'].item()
    i = 1
    for point_id in point_diffs:
        plt.plot(np.arange(len(point_diffs[point_id])), point_diffs[point_id], label=str(point_id))
        if i == 10 or point_id == len(point_diffs):
            plt.xlabel('rand times')
            plt.ylabel('diff')
            plt.legend()
            plt.show()
            i = 1
        i += 1

def read_inf_variance_acc_experiment_predict_distribution(read_configs):
    def ac_rate(file_name):
        point_diffs = np.load('{}/experiment_predict_distribution/{}'.format(read_configs['read_dir'], file_name), allow_pickle=True)['point_diffs'].item()
        ac_rate = np.zeros(100)
        for point_id in point_diffs:
            for i, diff in enumerate(point_diffs[point_id]):
                if diff > 0:
                    ac_rate[i] += 1
        return np.sort(ac_rate/len(point_diffs))
    
    ac = []
    
    precents = ['0.9', '0.7', '0.5', '0.3', '0.1']
    for precent in precents:
        ac.append(ac_rate('rand{}-lr-movielens.npz'.format(precent)))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    violin_plot(main_color='#D43F3A', line_color='k', scatter_color='white'
                , all_data=ac, x_axis_labels=precents, ax=ax,
                title='random samples accept rate',
                x_label='reserved percent',
                y_label='accept rate')
    plt.show()
    
def read_inf_variance_1by1_distribution(read_configs):
    sample_diffs = {}
    precents = ['0.9', '0.7', '0.5', '0.3', '0.1']
    for precent in precents:
        file_name = 'rand{}-lr-movielens.npz'.format(precent)
        point_diffs = np.load('{}/inf_variance_with_remains/{}'.format(read_configs['read_dir'], file_name), allow_pickle=True)['point_diffs'].item()
        for sample_id in point_diffs:
            if sample_id in sample_diffs.keys():
                sample_diffs[sample_id].append(point_diffs[sample_id])
            else:
                sample_diffs[sample_id] = [point_diffs[sample_id]]
    
    
    sample_id = 0
    for i in range(25):
        fig, axs = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)
        for row_axs in axs:
            for ax in row_axs:
                violin_plot(main_color='cyan', line_color='k', scatter_color='white'
                            , all_data=sample_diffs[sample_id], x_axis_labels=precents, ax=ax
                            , x_label='reserved precent'
                            , y_label='diff')
                sample_id += 1
        for ax in axs.flat:
            ax.label_outer()
        plt.show()

def read_inf_variance_distribution(read_configs):
    sample_mean_ac = {}
    precents = ['0.9', '0.7', '0.5', '0.3', '0.1']
    samples_diffs = {}
    for precent in precents:
        file_name = 'rand{}-lr-movielens.npz'.format(precent)
        point_diffs = np.load('{}/inf_variance_with_remains/{}'.format(read_configs['read_dir'], file_name), allow_pickle=True)['point_diffs'].item()
        for sample_id in point_diffs:
            ac_num = len([diff for diff in point_diffs[sample_id] if diff > 0])
            ac = ac_num/len(point_diffs[sample_id])
            if sample_id in sample_mean_ac.keys():
                sample_mean_ac[sample_id].append(ac)
            else:
                sample_mean_ac[sample_id] = [ac]
            if sample_id in samples_diffs.keys():
                samples_diffs[sample_id] += point_diffs[sample_id]
            else:
                samples_diffs[sample_id] = point_diffs[sample_id]
    
    mean_ac = [[], [], []]
    variance_ac = [[], [], []]
    diff_scatter = [[], [], []]
    for id in sample_mean_ac:
        mean = np.mean(np.array(sample_mean_ac[id]))
        variance = np.std(np.array(sample_mean_ac[id]))
        if mean < 0.25:
            mean_ac[0].append(mean)
            variance_ac[0].append(variance)
            diff_scatter[0].append(np.mean(samples_diffs[id]))
        elif mean > 0.75:
            mean_ac[2].append(mean)
            variance_ac[2].append(variance)
            diff_scatter[2].append(np.mean(samples_diffs[id]))
        else:
            mean_ac[1].append(mean)
            variance_ac[1].append(variance)
            diff_scatter[1].append(np.mean(samples_diffs[id]))
        
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    violin_plot(
        main_color='peru', line_color='k', scatter_color='white'
        , all_data=mean_ac, x_axis_labels=['<0.25', '0.25~0.75', '>0.75'], ax=ax1
        , title='mean'
        , x_label='accept rate'
        , y_label='mean accept rate')
    violin_plot(
        main_color='peru', line_color='k', scatter_color='white'
        , all_data=variance_ac, x_axis_labels=[len(variance_ac[0]), len(variance_ac[1]), len(variance_ac[2])], ax=ax2
        , title='Variance'
        , x_label='sample number of certain accept rate range'
        , y_label='variance')
    plt.show()
    
    plt.scatter(x=mean_ac[0], y=diff_scatter[0], color='cornflowerblue', label='<25%')
    plt.scatter(x=mean_ac[1], y=diff_scatter[1], color='slategrey', label='25%~75%')
    plt.scatter(x=mean_ac[2], y=diff_scatter[2], color='lightcoral', label='>75%')
    plt.xlabel('mean accept rate')
    plt.ylabel('predict diff')
    plt.legend()
    plt.show()

def read_experiment_possible_higher_accuracy(read_configs):
    performance = []
    percents = ['0.9', '0.7', '0.5', '0.3', '0.1']
    for percent in percents:
        file_name = 'perform_better{}-lr-fraud_detection.npz'.format(percent)
        file = np.load('{}/performance_higher_accuracy/{}'.format(read_configs['read_dir'], file_name), allow_pickle=True)['performance'].item()
        for eva_type in file:
            for acc in file[eva_type]:
                performance.append([percent, acc, eva_type])
    performance = pd.DataFrame(data=performance, columns=['remains precent', 'accuracy', 'data type'])
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax = sns.violinplot(x='remains precent', y='accuracy', hue='data type', data=performance, palette='muted', split=True)
    ax.axhline(y=0.7985, color='orangered', linestyle=':', label='accuracy with test data')
    ax.axhline(y=0.789, color='c', linestyle=':', label='accuracy with valid data')
    plt.legend()
    plt.show()

def read_experiment_small_model_select_points(read_configs):
    
    performance = np.load('{}/{}/{}'.format(read_configs['read_dir'], 'experiment_small_model_select_points', 'census_income_0.1rand_0.1select.npz'), allow_pickle=True)['performance'].item()
    performance = [performance['test'], performance['valid']]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    violin_plot(main_color='darkorange', line_color='k', scatter_color='white'
                , all_data=performance, ax=ax, x_axis_labels=['test', 'validation']
                , x_label='census_income', y_label='accuracy', title='small_select_model')
    ax.axhline(y=0.8071649712516585, color='orangered', linestyle=':', label='accuracy with test data')
    ax.axhline(y=0.7997788833609729, color='c', linestyle=':', label='accuracy with validation data')
    plt.legend()
    plt.show()

def read_experiment_small_model_select_points_comparing_with_rand(read_configs):
    model = read_configs['model']
    basic_acc = {'test': 0.7995, 'valid': 0.789}
    all_acc = []
    higher_acc = []
    for eva_type in read_configs['performance_eva_sets']:
        for id in range(1, 101):
            checkpoint = 'rand{}_num600___churn_lr_wd1e-02_step30000'.format(id)
            # random model
            model.load_model(checkpoint)
            eva_x, eva_y = model.np2tensor(model.dataset[eva_type].get_batch())
            eva_diff = model.model(eva_x) - eva_y
            random_acc = len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff)
            all_acc.append(['random', random_acc, eva_type])
            # select model
            model.load_model('select' + checkpoint[4:])
            eva_diff = model.model(eva_x) - eva_y
            select_acc = len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff)
            all_acc.append(['select', select_acc, eva_type])
            if random_acc > basic_acc[eva_type]:
                higher_acc.append(['random', random_acc, eva_type])
                higher_acc.append(['select', select_acc, eva_type])
    
    all_acc = pd.DataFrame(data=all_acc, columns=['model type', 'accuracy', 'eva type'])
    higher_acc = pd.DataFrame(data=higher_acc, columns=['model type', 'accuracy', 'eva type'])
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax = sns.violinplot(x='model type', y='accuracy', hue='eva type',
                    data=all_acc, palette='Set2', split=True,
                    scale='count')
    ax.axhline(y=basic_acc['test'], color='orangered', linestyle=':', label='accuracy with test data')
    ax.axhline(y=basic_acc['valid'], color='c', linestyle=':', label='accuracy with valid data')
    plt.show()
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax = sns.violinplot(x='model type', y='accuracy', hue='eva type',
                    data=higher_acc, palette='Set2', split=True,
                    scale='count')
    ax.axhline(y=basic_acc['test'], color='orangered', linestyle=':', label='accuracy with test data')
    ax.axhline(y=basic_acc['valid'], color='c', linestyle=':', label='accuracy with valid data')
    plt.show()
    
    select_test = np.array([row['accuracy'] for index, row in higher_acc.iterrows() \
                if row['model type'] == 'select' and row['eva type'] == 'test'])
    select_valid = np.array([row['accuracy'] for index, row in higher_acc.iterrows() \
                if row['model type'] == 'select' and row['eva type'] == 'valid'])
    
    
    rand_test = np.array([row['accuracy'] for index, row in higher_acc.iterrows() \
                if row['model type'] == 'random' and row['eva type'] == 'test'])
    rand_valid = np.array([row['accuracy'] for index, row in higher_acc.iterrows() \
                if row['model type'] == 'random' and row['eva type'] == 'valid'])

    plt.vlines(x=list(range(len(rand_test))), ymin=select_test, ymax=basic_acc['test'], linestyles='-', lw=3, colors='green', label='difference between big and random')
    plt.vlines(x=list(range(len(rand_test))), ymin=basic_acc['test'], ymax=rand_test, linestyles='-', lw=5, colors='orange', label='difference between big and select')
    plt.axhline(y=basic_acc['test'], color='orangered', linestyle=':', label='big accuracy')
    plt.title('test')
    plt.xlabel('model id')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.vlines(x=list(range(len(rand_valid))), ymin=select_valid, ymax=basic_acc['valid'], linestyles='-', lw=3, colors='green', label='difference between big and random')
    plt.vlines(x=list(range(len(rand_valid))), ymin=basic_acc['valid'], ymax=rand_valid, linestyles='-', lw=5, colors='orange', label='difference between big and select')
    plt.axhline(y=basic_acc['valid'], color='orangered', linestyle=':', label='big accuracy')
    plt.title('validation')
    plt.xlabel('model id')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

def read_select_points_distribution_on_rand_and_higher_accuracy_and_select(read_configs):
    model = read_configs['model']
    acc_baseline = {'test': 0.7995, 'valid': 0.789}
    all_rand_ids = np.array([])
    all_select_ids = np.array([])
    num_higher_rand_model = 0
    all_higher_rand_ids = np.array([])
    all_higher_select_ids = np.array([])

    eva_type = 'test'
    for model_id in range(100, 0, -1):
        # random model
        model.load_model("rand{}_num600___churn_lr_wd1e-02_step30000".format(model_id))
        all_rand_ids = np.append(all_rand_ids, model.remain_ids)
        
        eva_x, eva_y = model.np2tensor(model.dataset[eva_type].get_batch())
        eva_diff = model.model(eva_x) - eva_y
        random_acc = len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff)
        if random_acc > acc_baseline[eva_type]:
            num_higher_rand_model += 1
            all_higher_rand_ids = np.append(all_higher_rand_ids, model.remain_ids)
            

        # select model
        model.load_model("select{}_num600___churn_lr_wd1e-02_step30000".format(model_id))
        all_select_ids = np.append(all_select_ids, model.remain_ids)

        if random_acc > acc_baseline[eva_type]: # select ids correspond to random model which has higher accuracy
            all_higher_select_ids = np.append(all_higher_select_ids, model.remain_ids)

    infs = np.array([])
    # big model influence value
    for i in range(1, 3):
        infs_file = r'c:\Users\Zheng\PycharmProjects\pythonProject\research\influence impact\pytorchGenericNet\experiment_save_results\churn_infs\{}-2.npz'.format(i)
        infs = np.append(infs, np.load(infs_file)['infs'])

    dist = {}
    dist['sample id'] = np.arange(model.dataset['train'].x.shape[0], 0, -1)
    dist['random-all'] = np.array([])
    dist['select-all'] = np.array([])
    dist['random-higher'] = np.array([])
    dist['select-higher'] = np.array([])
    for sample_id in dist['sample id']:
        dist['random-all'] = np.append(dist['random-all'], np.count_nonzero(all_rand_ids == sample_id)/100)
        dist['select-all'] = np.append(dist['select-all'], np.count_nonzero(all_select_ids == sample_id)/100)
        dist['random-higher'] = np.append(dist['random-higher'], np.count_nonzero(all_higher_rand_ids == sample_id)/num_higher_rand_model)
        dist['select-higher'] = np.append(dist['select-higher'], np.count_nonzero(all_higher_select_ids == sample_id)/num_higher_rand_model)
    
    # sorted_id = np.argsort(-dist['random-higher'])
    sorted_id = np.argsort(-infs)
    infs = infs[sorted_id]
    dist['random-all'] = dist['random-all'][sorted_id]
    dist['select-all'] = dist['select-all'][sorted_id]
    dist['random-higher'] = dist['random-higher'][sorted_id]
    dist['select-higher'] = dist['select-higher'][sorted_id]

    for type in ['random-all', 'select-all', 'random-higher', 'select-higher']:
        plt.plot(dist[type], dist['sample id'])
        plt.title(type)
        plt.ylabel('sample id')
        plt.xlabel('sample point frequency')
        plt.show()
    
    plt.plot(infs, dist['sample id'])
    plt.title('big model')
    plt.ylabel('sample id')
    plt.xlabel('influence value')
    plt.show()


read_configs = set_configs()
# read_experiment_get_correlation(read_configs)
# read_remove_all_negtive(read_configs)
# read_inf_variance_change_with_randTime(read_configs)
# read_inf_variance_acc_experiment_predict_distribution(read_configs)
# read_inf_variance_1by1_distribution(read_configs)
# read_inf_variance_distribution(read_configs)
# read_experiment_possible_higher_accuracy(read_configs)
# read_experiment_small_model_select_points(read_configs)
# read_experiment_small_model_select_points_comparing_with_rand(read_configs)
read_select_points_distribution_on_rand_and_higher_accuracy_and_select(read_configs)