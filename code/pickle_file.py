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

        # print len(set_x), set_x[0].shape

        data_x = numpy.asarray(set_x, dtype=theano.config.floatX)
        data_y = numpy.asarray(set_y, dtype='int32')
        print (("File %s is loaded") % (read_name))
        return data_x, data_y

    # get shared variables of zeros
    @staticmethod
    def shared_zeros(image_size, batch_size):
        image_zeros = [numpy.zeros(image_size * image_size)] * batch_size
        label_zeros = [0] * batch_size
        
        # print len(image_zeros), image_zeros[0].shape
        
        # change to share data set
        shared_x = theano.shared(numpy.asarray(image_zeros, 
            dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(numpy.asarray(label_zeros, 
            dtype='int32'), borrow=True)
        return shared_x, shared_y

def test():
    # PickleFile.read_file("data/train_pickle", "train", 0)
    PickleFile.shared_zeros(48, 500)

if __name__ == '__main__':
    test()