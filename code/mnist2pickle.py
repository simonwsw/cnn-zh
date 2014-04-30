import os
import glob
import cPickle
import gzip

class Mnist2Pickle(object):
    def __init__(self):
        pass

    # read the pickle file from mnist
    @staticmethod
    def read_mnist(read_dir_name, read_file_name):
        read_name = os.path.join(os.path.split(__file__)[0], "..", 
            read_dir_name, read_file_name)
        with gzip.open(read_name, 'rb') as f:
            train_set, valid_set, test_set = cPickle.load(f)

        print ("File %s is read" % (read_file_name))
        return train_set, valid_set, test_set

    @staticmethod
    def write_pickle(write_dir_name, write_prefix, x_set, batch_size):
        batch_num = len(x_set[0]) / batch_size
        for i in range(batch_num):
            Mnist2Pickle.write_single_pickle(write_dir_name, write_prefix, 
                [x_set[0][i * batch_size: (i + 1) * batch_size], 
                x_set[1][i * batch_size: (i + 1) * batch_size]], i)

        print ("%i files are written to %s" % (batch_num, write_dir_name))
        
    @staticmethod
    def write_single_pickle(write_dir_name, write_prefix, x_set_batch, 
            batch_count):
        write_file_name = (("%s(%i).pkl.gz") % (write_prefix, batch_count))
        write_name = os.path.join(os.path.split(__file__)[0], "..", 
            write_dir_name, write_file_name)
        with gzip.open(write_name, 'wb') as f:
            cPickle.dump(x_set_batch, f, protocol=cPickle.HIGHEST_PROTOCOL)

        print ("File %s is written" % (write_file_name))


def main():
    train_set, valid_set, test_set = Mnist2Pickle.read_mnist("data", 
        "mnist.pkl.gz")
    Mnist2Pickle.write_pickle("data/m_train_pickle", "train", train_set, 500)
    Mnist2Pickle.write_pickle("data/m_valid_pickle", "valid", valid_set, 500)
    Mnist2Pickle.write_pickle("data/m_test_pickle", "test", test_set, 500)

def test():
    pass

if __name__ == '__main__':
    # test()
    main()
