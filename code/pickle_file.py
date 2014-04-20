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
    @staticmethod
    def read_file(read_dir, read_prefix, batch_count):
        read_name = (("%s(%i).pkl.gz") % (read_prefix, batch_count))
        open_name = os.path.join(os.path.split(__file__)[0], "..", read_dir, 
            read_name)
        with gzip.open(open_name, 'rb') as f:
            set_x, set_y = cPickle.load(f)

        print len(set_x), set_x[0].shape

        # change to share data set
        shared_x = theano.shared(numpy.asarray(set_x, 
            dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(numpy.asarray(set_y, 
            dtype=theano.config.floatX), borrow=True)

        print (("File %s is loaded") % (read_name))
        return shared_x, T.cast(shared_y, 'int32')

def test():
    PickleFile.read_file("data/train_pickle", "train", 0)

if __name__ == '__main__':
    test()