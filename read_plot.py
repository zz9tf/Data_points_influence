from cProfile import label
import numpy as np
import matplotlib.pyplot as plt


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

def read_rand_violin():
    def ac_rate(file_name):
        point_diffs = np.load('plot_result/{}'.format(file_name), allow_pickle=True)['point_diffs'].item()
        ac_rate = np.zeros(100)
        for point_id in point_diffs:
            for i, diff in enumerate(point_diffs[point_id]):
                if diff > 0:
                    ac_rate[i] += 1
        return np.sort(ac_rate/len(point_diffs))
    
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
        ax.set_xlabel('Sample name')
    
    ac = []
    precents = ['0.9', '0.7', '0.5', '0.3', '0.1']
    for precent in precents:
        ac.append(ac_rate('rand{}-lr-movielens.npz'.format(precent)))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    parts = ax.violinplot(ac, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    quartile1, medians, quartile3 = np.percentile(ac, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(data, q1, q3)
        for data, q1, q3 in zip(ac, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    
    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    set_axis_style(ax, precents)
    ax.set_xlabel("reserved percent")
    ax.set_ylabel("accept rate")
    ax.set_title('random samples accept rate')
    plt.show()
    

# read_diff()
# read_remove_all_negtive()
# read_rand()
read_rand_violin()