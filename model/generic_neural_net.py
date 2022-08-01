import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from model.symple_mf import MF
from model.NCF import NCF
from model.symple_lr import lr

class Model():

    def __init__(self, **kwargs):
        basic_configs = kwargs.pop('basic_configs')
        
        # loading data
        self.dataset = basic_configs.pop('dataset')
        self.remain_ids = np.array(range(self.dataset['train'].num_examples))

        # Use CUDA
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        if basic_configs['model'] == 'MF':
            self.model = MF(model_configs=kwargs.pop('model_configs'))
        elif basic_configs['model'] == 'NCF':
            self.model = NCF(model_configs=kwargs.pop('model_configs'))
        elif basic_configs['model'] == 'lr':
            self.model = lr(model_configs=kwargs.pop('model_configs'))
        else:
            assert NotImplementedError

        # training hyperparameter
        self.batch_size = basic_configs.pop('batch_size', None)
        self.learning_rate = basic_configs.pop('learning_rate')
        self.weight_decay = basic_configs.pop('weight_decay', None)

        # function for train
        self.loss_fn = self.model.get_loss_fn()
        self.optimizer = self.model.get_optimizer(self.model.parameters(), self.learning_rate)

        # save space for model performance: train_loss, test_loss
        self.performance = {}

        # create loading and saving location
        self.checkpoint_dir = basic_configs.pop('checkpoint_dir', 'checkpoint')
        if os.path.exists(self.checkpoint_dir) is False:
            os.makedirs(self.checkpoint_dir)
        self.model_name = basic_configs.pop('model_name')
        print(self.__str__(self.model.__str__()))

    def __str__(self, details):
        return 'Model name: {}\n'.format(self.model_name)\
               + '\nweight decay: {}\n'.format(self.weight_decay) \
               + 'number of training examples: {}\n'.format(self.dataset['train'].num_examples) \
               + 'number of testing examples: {}\n'.format(self.dataset['test'].num_examples) \
               + 'number of valid examples: {}\n'.format(self.dataset['valid'].num_examples) \
               + details\
               + '\n-------------------------------\n'

    def reset_train_dataset(self, remain_ids=None):
        if remain_ids is None:
            # recover to origin dataset
            self.remain_ids = np.array(range(self.dataset['train'].x.shape[0]))
        else:
            self.remain_ids = remain_ids
        
        
        self.dataset['train'].reset_using(self.remain_ids)

    def load_model(self, checkpoint_name=None):
        '''This method initializes model perameters or loads model checkpoint by 
        checkpoint_name. If the model dosen't exist, this method require user to
        input a valid checkpoint name or q to reset model parameter without load
        any model.

        Args:
            checkpoint_name (_type_, optional): The name of the model checkpoint. 
            Defaults to None.

        Returns:
            num_epoch: The epoch the model has been trained. So the model could
            continue its train with higher epochs.
        '''
        self.model.reset_parameters()
        num_epoch = 0
        all_files = os.listdir(os.path.join(self.checkpoint_dir))
        if checkpoint_name not in all_files and checkpoint_name != 'q':
            for file in all_files:
                print(file)
            checkpoint_name = input('Which model do you want to load?(q to exit)')
        while checkpoint_name not in all_files and checkpoint_name != 'q':
            for file in all_files:
                print(file)
            print('Please an input valid model name or type q to quit!')
            checkpoint_name = input('Which model do you want to load?(q to exit)')
        if checkpoint_name != 'q':
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name))
            num_epoch = checkpoint['epoch']
            self.reset_train_dataset(checkpoint['remain_ids'])
            self.performance = checkpoint['performance']
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'train_loss' not in self.performance.keys():
            self.performance['train_loss'] = [[]]
        if 'test_loss' not in self.performance.keys():
            self.performance['test_loss'] = [[]]

        return num_epoch
        

    def init_model(self, load_checkpoint=False, checkpoint_name=None):
        '''This method initializes model perameters or loads model checkpoint by 
        checkpoint_name. All checkpoint files should be stored in the path of 
        os.path.join(self.checkpoint_dir).

        Args:
            load_checkpoint (bool, optional): Whether load data. Defaults to False.
            checkpoint_name (_type_, optional): The name of the model checkpoint. 
            Defaults to None.
        '''
        self.model.reset_parameters()
        num_epoch = 0
        if load_checkpoint and checkpoint_name in os.listdir(os.path.join(self.checkpoint_dir)):
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name))
            num_epoch = checkpoint['epoch']
            self.reset_train_dataset(checkpoint['remain_ids'])
            self.performance = checkpoint['performance']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print('Load successfully.')
        if 'train_loss' not in self.performance.keys():
            self.performance['train_loss'] = [[]]
        if 'test_loss' not in self.performance.keys():
            self.performance['test_loss'] = [[]]

        return num_epoch
    
    def np2tensor(self, np_data):
        return [torch.tensor(data).to(self.device) for data in np_data]

    def train(self, num_epoch=180000, load_checkpoint=False, save_checkpoint=False, verbose=False,
            checkpoint_name='', plot=False):
        checkpoint_name += '___' + self.model_name + '_step%d' % num_epoch
        start = self.init_model(load_checkpoint, checkpoint_name)
        self.model.train()
        
        if verbose:
            print('--- Start {} ---'.format(checkpoint_name))
            print('\nTraining from {} to {} epoch'.format(start, num_epoch))
        else:
            print('processing: {} ({} epoch)'.format(checkpoint_name, num_epoch))

        for epoch in range(start, num_epoch):
            train_x, train_y = self.np2tensor(self.dataset['train'].get_batch(self.batch_size))
            train_predict = self.model(train_x)
            train_y = train_y.float()
            train_loss = self.loss_fn(train_predict, train_y)
            test_x, test_y = self.np2tensor(self.dataset['test'].get_batch(self.batch_size))
            test_predict = self.model(test_x)
            test_loss = self.loss_fn(test_predict, test_y)
            
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            if verbose and epoch % 1000 == 0:
                print('Epoch {}: loss = {:.8f}'.format(epoch, train_loss.item()))

            self.performance['train_loss'][-1].append(train_loss.item())
            self.performance['test_loss'][-1].append(test_loss.item())

        if save_checkpoint and checkpoint_name not in os.listdir(os.path.join(self.checkpoint_dir)):
            torch.save({
                'epoch': num_epoch,
                'remain_ids': self.remain_ids,
                'performance': self.performance,
                'model_state_dict': self.model.state_dict()
                }, os.path.join(self.checkpoint_dir, checkpoint_name))
        
        if plot:
            self.plot_loss_of_train_process()
        return checkpoint_name

    def plot_loss_of_train_process(self):
        '''This method prints a plot of all loss during train process
        '''
        plt.plot(self.performance['train_loss'][-1], label='train loss')
        plt.plot(self.performance['test_loss'][-1], label='test loss')
        plt.legend()
        plt.show()

    def evaluate(self):
        train_x, train_y = self.np2tensor(self.dataset['train'].get_batch())
        train_perdict = self.model(train_x)
        train_loss = self.loss_fn(train_perdict, train_y)
        grads = torch.autograd.grad(train_loss, self.model.parameters())

        test_x, test_y = self.np2tensor(self.dataset['test'].get_batch())
        test_predict = self.model(test_x)
        test_loss = self.loss_fn(test_predict, test_y)
        print('\nEvaluation:')
        print('Train loss on all data: {}'.format(train_loss.item()))
        print('Train acc on all data: {}'.format(torch.mean(1 - torch.abs(train_perdict - train_y))))

        print('Test loss on all data: {}'.format(test_loss))
        print('Test acc on all data: {}'.format(torch.mean(1 - torch.abs(test_predict - test_y))))

        gradient = torch.cat([torch.flatten(grad) for grad in grads])
        print('Norm of the mean of gradients {}'.format(torch.linalg.norm(gradient)))
