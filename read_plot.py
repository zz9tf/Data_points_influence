import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_diff():
    diff_dic = np.load("plot_result/result-lr-movielens.npz")

    plt.scatter(diff_dic["real_diffs"], diff_dic["pred_diffs"])
    all_min = min(min(diff_dic["real_diffs"]), min(diff_dic["pred_diffs"]))
    all_max = max(max(diff_dic["real_diffs"]), max(diff_dic["pred_diffs"]))
    plt.plot([all_min, all_max], [all_min, all_max], color="blue", label="x=y")
    real_min = np.min(diff_dic["real_diffs"])
    real_max = np.max(diff_dic["real_diffs"])
    predict_min = np.min(diff_dic["pred_diffs"])
    predict_max = np.max(diff_dic["pred_diffs"])
    plt.plot([real_min, real_max], [predict_min, predict_max], color="red", label="min2max on two D")
    plt.plot([real_min, real_max], [0, 0], color="orange", label="predict effect dividing line")
    plt.plot([0, 0], [predict_min, predict_max], color="orange", label="real effect dividing line")
    plt.xlabel("real diff")
    plt.ylabel("predict diff")
    plt.legend()
    plt.show()

def read_remove_all_negtive():
    remove_all_dic = np.load('plot_result/remove_all_negtive-lr-movielens.npz', allow_pickle=True)
    print(remove_all_dic['ids'])
    samples_diff_dic = remove_all_dic['samples_diff_dic'].item()
    for item in samples_diff_dic.items():
        plt.plot(np.arange(len(item[1])), item[1], label=str(item[0]))
        plt.xlabel("num of remove points")
        plt.ylabel("diff")
        plt.legend()
        plt.show()

def read_rand():
    point_diffs = np.load('plot_result/rand0.1-lr-movielens.npz', allow_pickle=True)['point_diffs'].item()
    i = 1
    for point_id in point_diffs:
        plt.plot(np.arange(len(point_diffs[point_id])), point_diffs[point_id], label=str(point_id))
        if i == 10 or point_id == len(point_diffs):
            plt.xlabel("rand times")
            plt.ylabel("diff")
            plt.legend()
            plt.show()
            i = 1
        i += 1

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

def read_rand_violin():
    def ac_rate(file_name):
        point_diffs = np.load('plot_result/{}'.format(file_name), allow_pickle=True)['point_diffs'].item()
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
    
def read_adjust_violin1():
    sample_diffs = {}
    precents = ['0.9', '0.7', '0.5', '0.3', '0.1']
    for precent in precents:
        file_name = 'rand{}-lr-movielens.npz'.format(precent)
        point_diffs = np.load('plot_result/{}'.format(file_name), allow_pickle=True)['point_diffs'].item()
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

def read_adjust_violin2():
    sample_mean_ac = {}
    precents = ['0.9', '0.7', '0.5', '0.3', '0.1']
    samples_diffs = {}
    for precent in precents:
        file_name = 'rand{}-lr-movielens.npz'.format(precent)
        point_diffs = np.load('plot_result/{}'.format(file_name), allow_pickle=True)['point_diffs'].item()
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
    plt.xlabel("mean accept rate")
    plt.ylabel("predict diff")
    plt.legend()
    plt.show()

def read_perform_better():
    performance = []
    percents = ['0.9', '0.7', '0.5', '0.3', '0.1']
    for percent in percents:
        file_name = 'perform_better{}-lr-movielens.npz'.format(percent)
        file = np.load('plot_result/{}'.format(file_name), allow_pickle=True)['performance'].item()
        for eva_type in file:
            for acc in file[eva_type]:
                performance.append([percent, acc, eva_type])
    performance = pd.DataFrame(data=performance, columns=['remains precent', 'accuracy', 'data type'])
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # violin_plot(main_color='darkorange', line_color='k', scatter_color='white'
    #             , all_data=performance, x_axis_labels=precents, ax=ax
    #             , x_label="remains precent", y_label="accuracy", title="less data, better accuracy")
    ax = sns.violinplot(x="remains precent", y="accuracy", hue="data type", data=performance, palette="muted", split=True)
    ax.axhline(y=0.2152317880794702, color='orangered', linestyle=':', label="accuracy with test data")
    ax.axhline(y=0.21928807947019868, color='c', linestyle=':', label="accuracy with valid data")
    plt.legend()
    plt.show()


# read_diff()
# read_remove_all_negtive()
# read_rand()
# read_rand_violin()
# read_adjust_violin1()
# read_adjust_violin2()
read_perform_better()
