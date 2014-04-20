import time

import numpy
import theano
import theano.tensor as T

from pickle_file import PickleFile
from conv_module import LogisticRegression, HiddenLayer
from conv_module import LeNetConvPoolLayer, LeNetConvPoolParam

def lenet():
    
    # set up parameters
    class_count = 3755
    train_dir = "data/train_pickle"
    train_prefix = "train"
    valid_dir = "data/valid_pickle"
    valid_prefix = "valid"
    test_dir = "data/test_pickle"
    test_prefix = "test"
    n_train_batches = 300
    n_valid_batches = 100
    n_test_batches = 100
    batch_size = 500
    learning_rate=0.1
    image_size = 48

    # set up random number
    rng = numpy.random.RandomState(23455)

    # initialize shared variables
    train_set_x, train_set_y = PickleFile.shared_zeros(image_size, batch_size)
    valid_set_x, valid_set_y = PickleFile.shared_zeros(image_size, batch_size)
    test_set_x, test_set_y = PickleFile.shared_zeros(image_size, batch_size)

    # ========== build the model ==========
    # start to build the model. prepare the parameters
    print "Building the model..."
    layer0_param = LeNetConvPoolParam(input_image_size=48, 
        input_feature_num=1, filter_size=5, pooling_size=2, kernel=10)
    layer1_param = LeNetConvPoolParam(
        input_image_size=layer0_param.output_size, 
        input_feature_num=layer0_param.kernel, 
        filter_size=5, pooling_size=2, kernel=20)
    layer2_output_size = 100
    layer3_output_size = class_count

    # allocate symbolic variables for the data: 
    # mini batch index, rasterized images, labels
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    # seshape matrix of rasterized images of shape 
    # (batch_size, image_size ** 2)
    # to a 4d tensor, compatible with LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, layer0_param.input_feature_num, 
        layer0_param.input_image_size, layer0_param.input_image_size))

    # construct the first convolutional pooling layer:
    # filtering reduces the image size to: 
    # ((image_size - conv_size + 1) ** 2)
    # maxpooling reduces this further to: ((conv_output / max) ** 2)
    # 4d output tensor is thus of shape: (batch_size, kernel, output ** 2)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
        image_shape=(batch_size, layer0_param.input_feature_num, 
            layer0_param.input_image_size, layer0_param.input_image_size),
        filter_shape=(layer0_param.kernel, 
            layer0_param.input_feature_num, layer0_param.filter_size, 
            layer0_param.filter_size), 
        poolsize=(layer0_param.pooling_size, 
            layer0_param.pooling_size))

    # construct the second convolutional pooling layer
    # 4d output tensor is thus of shape: (batch_size, kernel, output ** 2)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
        image_shape=(batch_size, layer1_param.input_feature_num, 
            layer1_param.input_image_size, layer1_param.input_image_size),
        filter_shape=(layer1_param.kernel, 
            layer1_param.input_feature_num, layer1_param.filter_size, 
            layer1_param.filter_size), 
        poolsize=(layer1_param.pooling_size, 
            layer1_param.pooling_size))

    # the hiddenLayer being fully-connected, 
    # it operates on 2d matrices of shape: (batch_size, num_pixels)
    # this will generate a matrix of shape: 
    # (batch_size, kernel[1] * output ** 2)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, 
        n_in=(layer1_param.kernel * layer1_param.output_size * 
            layer1_param.output_size), n_out=layer2_output_size, 
        activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, 
        n_in=layer2_output_size, n_out=layer3_output_size)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train model updates set up with list
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([], layer3.errors(y), 
        givens={x: test_set_x, y: test_set_y})

    validate_model = theano.function([], layer3.errors(y),
        givens={x: valid_set_x, y: valid_set_y})

    # train_model is a function that updates the model parameters by sgd
    train_model = theano.function([], cost, updates=updates,
        givens={x: train_set_x, y: train_set_y})


    # ========== train model ==========
    # prepare the parameters
    print "Training the model..."
    n_epochs = 200
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995

    # time to check valid set after how many minibatches
    validation_frequency = min(n_train_batches, patience / 2)

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    # start to loop
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index

            # print training iteration and do training
            print "training @ iter = ", iter
            
            # load new train data
            new_train_set_x, new_train_set_y = PickleFile.read_file(
                train_dir, train_prefix, minibatch_index)
            train_set_x.set_value(new_train_set_x, borrow=True)
            train_set_y.set_value(new_train_set_y, borrow=True)
            cost_ij = train_model()

            # compute zero-one loss on validation set
            if (iter + 1) % validation_frequency == 0:
                validation_losses = []
                
                for i in xrange(n_valid_batches):
                    # load new valid data
                    new_valid_set_x, new_valid_set_y = PickleFile.read_file(
                        valid_dir, valid_prefix, i)
                    valid_set_x.set_value(new_valid_set_x, borrow=True)
                    valid_set_y.set_value(new_valid_set_y, borrow=True)
                    validation_losses.append(validate_model())

                this_validation_loss = numpy.mean(validation_losses)
                print ("Epoch %i, batch %i/%i, validation error %f %%" % 
                    (epoch, minibatch_index + 1, n_train_batches, 
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < (best_validation_loss * 
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = []

                    for i in xrange(n_test_batches):
                        # load new test data
                        new_test_set_x, new_test_set_y = PickleFile.read_file(
                            test_dir, test_prefix, i)
                        test_set_x.set_value(new_test_set_x, borrow=True)
                        test_set_y.set_value(new_test_set_y, borrow=True)
                        test_losses.append(test_model())

                    test_score = numpy.mean(test_losses)
                    print (("    -> test error of best model %f %%") %
                        (test_score * 100.))

            # reach the patience and end loop
            if patience <= iter:
                done_looping = True
                break
    
    # finish run
    end_time = time.clock()
    print "Optimization complete."
    print (("Best validation score of %f %% obtained at iteration %i") % 
        (best_validation_loss * 100.0, best_iter + 1))
    print (("    -> with test performance %f %%") % (test_score * 100.0))
    print >> sys.stderr, ("The code for file " + 
        os.path.split(__file__)[1] + 
        " ran for %.2fm" % ((end_time - start_time) / 60.0))

if __name__ == '__main__':
    lenet()