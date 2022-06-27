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

# read_diff()
# read_remove_all_negtive()
read_rand()