import numpy as np


def load_movielens(dir):
    train = np.loadtxt("%s/ml-1m-ex.train.rating" % dir, delimiter='\t')
    valid = np.loadtxt("%s/ml-1m-ex.valid.rating" % dir, delimiter='\t')
    test = np.loadtxt("%s/ml-1m-ex.test.rating" % dir, delimiter='\t')

    train_input = train[:, :2].astype(np.int32)
    train_output = train[:, 2].astype(np.float32)
    valid_input = valid[:, :2].astype(np.int32)
    valid_output = valid[:, 2].astype(np.float32)
    test_input = test[:, :2].astype(np.int32)
    test_output = test[:, 2].astype(np.float32)

    train = Dataset(train_input, train_output)
    validation = Dataset(valid_input, valid_output)
    test = Dataset(test_input, test_output)

    return {"train": train, "validation": validation, "test": test}


def load_yelp(dir):
    train = np.loadtxt("%s/yelp-ex.train.rating" % dir, delimiter='\t')
    valid = np.loadtxt("%s/yelp-ex.valid.rating" % dir, delimiter='\t')
    test = np.loadtxt("%s/yelp-ex.test.rating" % dir, delimiter='\t')

    train_input = train[:, :2].astype(np.int32)
    train_output = train[:, 2].astype(np.float32)
    valid_input = valid[:, :2].astype(np.int32)
    valid_output = valid[:, 2].astype(np.float32)
    test_input = test[:, :2].astype(np.int32)
    test_output = test[:, 2].astype(np.float32)

    train = Dataset(train_input, train_output)
    validation = Dataset(valid_input, valid_output)
    test = Dataset(test_input, test_output)

    return {"train": train, "validation": validation, "test": test}


class Dataset(object):

    def __init__(self, x, y):

        assert (x.shape[0] == y.shape[0])

        self.x = x
        self.using_x = np.copy(x)
        self.y = y
        self.using_y = np.copy(y)
        self.num_examples = x.shape[0]
        self.index_in_epoch = 0

    def append_one_set(self, case_x, case_y):
        self.x = np.concatenate([self.x, case_x], axis=0)
        self.y = np.concatenate([self.y, case_y], axis=0)
        self.using_x = np.copy(self.x)
        self.using_y = np.copy(self.y)
        self.num_examples = self.x.shape[0]
        print("A set is added to the dataset.")
        return self.x.shape[0] - 1

    def reset_using(self, idxs=None):
        """
        This method will reset the samples in the dataset to be operated.
        :param idxs: A 1d numpy array include all indexs to be kept
        :param keep_idxs: A list includes all special indexs to be kept.
        :return: None
        """
        self.index_in_epoch = 0
        self.using_x = np.copy(self.x).take(idxs, axis=0)
        self.using_y = np.copy(self.y).take(idxs)
        self.num_examples = self.using_x.shape[0]

    def get_batch(self, batch_size=None):
        """
        This method will return a batch of data, and move index to the start of next batch
        :param batch_size: A integer represents the batch size
        :return: A tuple contains two tensor list which are batches of x and batches of y
        """
        if batch_size is None:
            return self.using_x, self.using_y

        if self.index_in_epoch >= self.num_examples:
            # # Shuffle the data
            # shuf_index = np.arange(self.num_examples)
            # np.random.shuffle(shuf_index)
            # self.using_x = self.using_x[shuf_index, :]
            # self.using_y = self.using_y[shuf_index]

            # Start next epoch
            self.index_in_epoch = 0

        start = self.index_in_epoch
        end = min(self.index_in_epoch + batch_size, self.num_examples)
        
        self.index_in_epoch += batch_size

        return self.using_x[start:end], self.using_y[start:end]

    def get_by_idxs(self, idxs=None):
        assert idxs is not None
        return self.x.take(idxs, axis=0).reshape((-1,2)), self.y.take(idxs)

    def get_related_idxs(self, x_idx):
        """This method returns related indexs of provided index.

        Args:
            idx (numpy int): represents the target index to find related indexes.

        Returns:
            all_id (numpy list): represents all related indexes of the provided index.
        """
        related_u_id = np.where(self.using_x[:, 0] == x_idx[0])[0]
        related_i_id = np.where(self.using_x[:, 1] == x_idx[1])[0]
        all_id = np.concatenate((related_u_id, related_i_id))

        return all_id
