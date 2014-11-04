import os
import sys
import numpy
from PIL import Image

def read_special_hex(f, length):
    num_hex_str = ""
    for i in range(length):
        hex_2b = f.read(1)
        num_hex_str = hex_2b + num_hex_str

    return int(num_hex_str.encode('hex'), 16)

def read_header(f):
    # get the length of the header
    header_length = read_special_hex(f, 4)
    print "Header length: " + str(header_length)

    # read format code
    format_code = f.read(8)
    
    # read illustration
    illustration = f.read(header_length - 62)
    print "Illustration: " + illustration

    # read code type and length
    code_type = read_special_hex(f, 20)
    code_length = read_special_hex(f, 2)

    # read data type
    data_type = f.read(20)
    print "Data type: " + data_type

    # read sample number
    sample_number = read_special_hex(f, 4)
    print "Sample number: " + str(sample_number)

    # read dimension
    dimension = read_special_hex(f, 4)
    print "Dimension: " + str(dimension)
    print "==== End of header ===="

    return sample_number

def display_sample_lable(f, sample_id):
    lable_gb = f.read(2)
    
    if lable_gb.encode('hex') is 'ff':
        print  "Sample " + str(sample_id) + ": garbage :-("
        return True, None
    else:
        lable_uft8 = lable_gb.decode('gb18030').encode('utf-8')
        print "Sample " + str(sample_id) + ": " + lable_uft8
        return False, lable_uft8

def display_sample_image(f, sample_id):
    image_1d = []
    for i in xrange(512):
        pixel = f.read(1)
        image_1d.append(int(pixel.encode('hex'), 16))

    #print image_1d

def display_sample(f, sample_id):
    garbage, lable_uft8 = display_sample_lable(f, sample_id)
    if garbage:
        pass
    else:
        display_sample_image(f, sample_id)

def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == dataset:
            dataset = new_path

    # Load the dataset
    with open(dataset, 'rb') as f:
        sample_number = read_header(f)
        sample_number = 3747

        # read samples
        #for sample_id in xrange(sample_number):
        #   display_sample(f, sample_id)

def show_info(train_set, valid_set, test_set):
    # show shape of the set
    print (("train_set: %i rows, %i columns") % (len(train_set), len(train_set[0])))
    print (("valid_set: %i rows, %i columns") % (len(valid_set), len(valid_set[0])))
    print (("test_set: %i rows, %i columns") % (len(test_set), len(test_set[0])))

def select_image(train_set, valid_set, test_set, col=28, number=10):
    for i in range(number):
        # reshape the array
        adjust = numpy.reshape(train_set[0][i], (-1, col))
        save_image(adjust, train_set[1][i], i)

def save_image(matrix, number, number_id):
    im = Image.fromarray(matrix)
    name = ("tmp/test-%i (%i).tiff") % (number, number_id)
    im.save(name)

def display_char():
    # get the data
    load_data("1241.mpf")

    # print data information
    #show_info(train_set, valid_set, test_set)

    # save to image
    #select_image(train_set, valid_set, test_set)

if __name__ == '__main__':
    display_char()