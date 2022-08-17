import time
from functools import reduce
import numpy as np
import os
import torch
from configs.loading_configs import loading_args
from lib.load_data import load_data
import lib.custom_methods as custom_methods
from model.generic_neural_net import Model

def secondsToStr(t):
    return '%d:%02d:%02d.%03d' % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])

start_time = time.time()
print(time.strftime('Start time: %H:%M:%S', time.localtime()))

args = loading_args()
for key in vars(args):
    print(key, args[key])

dataset = load_data(os.path.join(args.datapath, args.dataset))

model = Model(
    # model
    model_configs=args.model_configs[args.model],
    basic_configs={
        'model': args.model,
        # loading data
        'dataset': dataset,
        # train configs
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        
        # loading configs
        'checkpoint_dir': args.checkpoint_dir,
        'model_name': '%s_%s_wd%.0e' % (
            args.dataset, args.model, args.weight_decay)
    }
)

if args.task == None:
    print("Please input a task to excuse")
elif args.task =='test':
    model.train(verbose=True, plot=True)
    eva_x, eva_y = model.np2tensor(model.dataset['test'].get_batch())
    eva_diff = model.model(eva_x) - eva_y
    print(len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff))
elif args.task == 'experiment_get_correlation':
    custom_methods.experiment_get_correlation(
        model=model,
        configs=args
    )

elif args.task == 'experiment_predict_distribution':
    custom_methods.experiment_predict_distribution(
        model=model,
        configs=args,
        precent_to_keep=0.7
    )

elif args.experiment == 'experiment_possible_higher_accuracy':
    custom_methods.experiment_possible_higher_accuracy(
        model=model,
        configs=args,
        experiment_configs=args.experiment_possible_higher_accuracy
    )

elif args.experiment == 'experiment_small_model_select_points':
    custom_methods.experiment_small_model_select_points(
        model=model,
        configs=args,
        experiment_configs=args.experiment_small_model_select_points
    )

elif args.experiment == 'experiment_small_model_based_big_model':
    custom_methods.experiment_small_model_based_big_model(
        model=model,
        configs=args,
        experiment_configs=args.experiment_small_model_based_big_model
    )

elif args.experiment == 'experiment_remove_all_negtive':
    custom_methods.experiment_remove_all_negtive(
        model=model,
        configs=args,
        experiment_configs=args.experiment_remove_all_negtive
    )
else:
    raise Exception('No such a task {}'.format(args.task))

end_time = time.time()
print(time.strftime('%H:%M:%S', time.localtime()))
print('Use time: {}'.format(secondsToStr(end_time - start_time)))