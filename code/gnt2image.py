import os
import glob
import numpy, scipy.misc
from PIL import Image

class SingleGntImage(object):
    def __init__(self, f):
        self.f = f

    def read_gb_label(self):
        label_gb = self.f.read(2)

        # check garbage label
        if label_gb.encode('hex') is 'ff':
            return True, None
        else:
            label_uft8 = label_gb.decode('gb18030').encode('utf-8')
            return False, label_uft8

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

class ReadGntFile(object):
    def __init__(self):
        pass

    def find_file(self):
        file_extend = ".gnt"
        self.file_list = []

        # get all gnt files in the dir
        dir_path = os.path.join(os.path.split(__file__)[0], "..", "data")
        for file_name in glob.glob(os.path.join(dir_path, '*.gnt')):
            self.file_list.append(file_name)

        return self.file_list

    def show_image(self):
        end_of_image = False
        count_file = 0
        count_single = 0
        width_list = []
        height_list = []

        #open all gnt files
        for file_name in self.file_list:
            count_file = count_file + 1
            with open(file_name, 'rb') as f:
                while not end_of_image:
                    count_single = count_single + 1
                    this_single_image = SingleGntImage(f)

                    # get the pixel matrix of a single image
                    label, pixel_matrix, width, height, end_of_image = \
                        this_single_image.read_single_image()
                    
                    width_list.append(width)
                    height_list.append(height)
                    print count_single, label, \
                        width, height, numpy.shape(pixel_matrix)

                    #self.save_image(pixel_matrix, label, count_single)
                    #if count_single >= 10:
                    #   end_of_image = True

            print ("End of file #%i") % (count_file)

    def save_image(self, matrix, label, count):
        im = Image.fromarray(matrix)
        name = ("tmp/test-%s (%i).tiff") % (label, count)
        im.save(name)

def display_char_image():
    gnt_file = ReadGntFile()
    file_list = gnt_file.find_file()
    gnt_file.show_image()

if __name__ == '__main__':
    display_char_image()