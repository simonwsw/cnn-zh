import os
import glob
import numpy, scipy.misc
import cPickle
import gzip

# shared dict and its values
utf8int_dict = {}
utf8int_max = 0

class SingleGntImage(object):
    def __init__(self, f):
        self.f = f

    def read_gb_label(self):
        global utf8int_dict, utf8int_max
        label_gb = self.f.read(2)

        # check garbage label
        if label_gb.encode('hex') is 'ff':
            return True, None
        else:
            label_utf8 = label_gb.decode('gb18030').encode('utf-8')

            # utf-8 to int
            if label_utf8 in utf8int_dict:
                return False, utf8int_dict[label_utf8]
            else:
                utf8int_dict[label_utf8] = utf8int_max
                utf8int_max += 1
                return False, utf8int_max - 1

    def read_special_hex(self, length):
        num_hex_str = ""
        
        # switch the order of bits
        for i in range(length):
            hex_2b = self.f.read(1)
            num_hex_str = hex_2b + num_hex_str

        return int(num_hex_str.encode('hex'), 16)

    def read_single_image(self):
        
        # zero-one value
        max_value = 255.0
        margin = 4

        # try to read next single image
        try:
            self.next_length = self.read_special_hex(4)
        except ValueError:
            print "Notice: end of file"
            return None, None, None, None, True

        # read the chinese utf-8 label
        self.is_garbage, self.label = self.read_gb_label()

        # read image width and height and do assert
        self.width = self.read_special_hex(2)
        self.height = self.read_special_hex(2)
        assert self.next_length == self.width * self.height + 10

        # read image matrix
        image_matrix_list = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(self.read_special_hex(1))

            image_matrix_list.append(row)

        # convert to numpy ndarray with size of 40 * 40 and add margin of 4
        self.image_matrix_numpy = \
            scipy.misc.imresize(numpy.array(image_matrix_list), \
            size=(40, 40)) / max_value
        self.image_matrix_numpy = numpy.lib.pad(self.image_matrix_numpy, \
            margin, self.padwithones)
        return self.label, self.image_matrix_numpy, \
            self.width, self.height, False

    def padwithones(self, vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = 1.0
        vector[-pad_width[1]:] = 1.0
        return vector

class GntFiles(object):
    def __init__(self, file_dir):
        self.file_dir = file_dir

    def find_file(self):
        file_extend = ".gnt"
        self.file_list = []

        # get all gnt files in the dir
        dir_path = os.path.join(os.path.split(__file__)[0], "..", 
            self.file_dir)
        for file_name in glob.glob(os.path.join(dir_path, '*.gnt')):
            self.file_list.append(file_name)

        return self.file_list

    def utf8int(self, label_utf8):
        # utf-8 to int
        if label_utf8 in self.utf8int_dict:
            return self.utf8int_dict[label_utf8]
        else:
            self.utf8int_max += 1
            self.utf8int_dict[label_utf8] = self.utf8int_max
            print label_utf8, ":", self.utf8int_dict[label_utf8]
        return self.utf8int_max

    def read_file(self, file_number_limit=None):
        global utf8int_max
        self.find_file()

        count_file = 0
        self.pickel_array = [[], []]
        print (("Reading %i files from %s directory...") % 
            (len(self.file_list), self.file_dir))
        
        #open all gnt files
        for file_name in self.file_list:
            count_file = count_file + 1
            with open(file_name, 'rb') as f:
                end_of_image = False
                count_single = 0
                while not end_of_image:
                    count_single = count_single + 1
                    this_single_image = SingleGntImage(f)

                    # get the pixel matrix of a single image
                    label, pixel_matrix, width, height, end_of_image = \
                        this_single_image.read_single_image()

                    # load matrix ato 1d feature to array
                    if not end_of_image:
                        self.pickel_array[0].append(pixel_matrix.reshape(-1))
                        self.pickel_array[1].append(label)

            print ("Finish file #%i with %i samples. Char count %i") % \
                (count_file, count_single - 1, utf8int_max)

            if not file_number_limit is None and (count_file >= 
                file_number_limit):
                print "File number reach limit of", file_number_limit
                break

        print count_file, "files are read"

    def write_file(self, file_dir, file_name):
        global utf8int_max
        save_name = os.path.join(os.path.split(__file__)[0], "..", file_dir, \
            file_name)

        # dump to pickle
        with gzip.open(save_name, 'wb') as f:
            cPickle.dump(self.pickel_array, f, 
                protocol=cPickle.HIGHEST_PROTOCOL)

        print (("File %s with %i classes is dumped") % 
            (file_name, utf8int_max))

    def save_image(self, matrix, label, count):
        im = Image.fromarray(matrix)
        name = ("tmp/test-%i (%i).tiff") % (label, count)
        im.save(name)

def gnt2pickle():
    # train set data
    train_gnt_files = GntFiles("data/train_set")
    train_gnt_files.read_file()
    train_gnt_files.write_file("data", "train.pkl.gz")

    # valid set data
    valid_gnt_files = GntFiles("data/valid_set")
    valid_gnt_files.read_file()
    valid_gnt_files.write_file("data", "valid.pkl.gz")

    # test set data
    test_gnt_files = GntFiles("data/test_set")
    test_gnt_files.read_file()
    test_gnt_files.write_file("data", "test.pkl.gz")

def test():
    # train set data
    train_gnt_files = GntFiles("data")
    train_gnt_files.read_file()

if __name__ == '__main__':
    gnt2pickle()
    # test()