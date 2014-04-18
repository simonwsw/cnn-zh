import os
import glob
import cPickle
import gzip

import numpy
import theano
import theano.tensor as T

class PickleFile(object):
    def __init__(self):
        pass

    # the return size is subjected to the batch size
    # not all the data in the file will be returned
    def read_file(self, file_dir, batch_size=None):

        # train set
        open_name = os.path.join(os.path.split(__file__)[0], "..", file_dir, \
            "train.pkl.gz")
        with gzip.open(open_name, 'rb') as f:
            pickle_array = cPickle.load(f)

        # regulize for batch
        if batch_size is None:
            n_train = len(pickle_array[0])
        else:
            n_train_batches = len(pickle_array[0]) / batch_size
            n_train = n_train_batches * batch_size

        train_set_x, train_set_y = self.shared_dataset(pickle_array[:][:n_train])
        print "File train.pkl.gz is loaded"

        # valid set
        open_name = os.path.join(os.path.split(__file__)[0], "..", file_dir, \
            "valid.pkl.gz")
        with gzip.open(open_name, 'rb') as f:
            pickle_array = cPickle.load(f)

        # regulize for batch
        if batch_size is None:
            n_valid = len(pickle_array[0])
        else:
            n_valid_batches = len(pickle_array[0]) / batch_size
            n_valid = n_valid_batches * batch_size
        valid_set_x, valid_set_y = self.shared_dataset(pickle_array[:][:n_valid])
        print "File valid.pkl.gz is loaded"

        # test set
        open_name = os.path.join(os.path.split(__file__)[0], "..", file_dir, \
            "test.pkl.gz")
        with gzip.open(open_name, 'rb') as f:
            pickle_array = cPickle.load(f)

        # regulize for batch
        if batch_size is None:
            n_test = len(pickle_array[0])
        else:
            n_test_batches = len(pickle_array[0]) / batch_size
            n_test = n_test_batches * batch_size
        print "File test.pkl.gz is loaded"
        test_set_x, test_set_y = self.shared_dataset(pickle_array[:][:n_test])

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        n_batches = [n_train_batches, n_valid_batches, n_test_batches]
        return rval, n_batches

    # use shared data to make faster copy between gpu and memory
    def shared_dataset(self, data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, \
            dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, \
            dtype=theano.config.floatX), borrow=borrow)
        
        # store data as int32 to make data copy more efficiently
        return shared_x, T.cast(shared_y, 'int32')

    def print_info(self):
        print len(self.pickle_array[0])
        print self.pickle_array[0][0].shape

def read_pickle():
    pickle_file = PickleFile()
    pickle_file.read_file("data")

if __name__ == '__main__':
    read_pickle()