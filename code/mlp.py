import time, datetime
import os, sys

import numpy
import theano
import theano.tensor as T

from pickle_file import PickleFile
from conv_module import LogisticRegression, HiddenLayer

def mlp():
    
    try:
        read_postfix = sys.argv[1]
        class_count = sys.argv[2]
        image_size = sys.argv[3]
        n_train_batches = sys.argv[4]
        n_valid_batches = sys.argv[5]
        n_test_batches = sys.argv[6]
    except IndexError:
        print "Usage: mlp.py postfix class image train# valid# test#"
        sys.exit(1)

    # set up parameters
    train_dir = "data/train_pickle" + read_postfix
    train_prefix = "train"
    valid_dir = "data/valid_pickle" + read_postfix
    valid_prefix = "valid"
    test_dir = "data/test_pickle" + read_postfix
    test_prefix = "test"
    
    batch_size = 500
    learning_rate = 0.1
    n_hidden = 500
    l1_reg=0.00
    l2_reg=0.0001

    # set up log file
    log_dir = "tmp"
    log_name = "log at " + str(datetime.datetime.now()) + ".txt"
    log_file = os.path.join(os.path.split(__file__)[0], "..", log_dir, 
        log_name)

    # set up random number
    rng = numpy.random.RandomState(23455)

    # initialize shared variables
    train_set_x, train_set_y = PickleFile.shared_zeros(image_size, batch_size)
    valid_set_x, valid_set_y = PickleFile.shared_zeros(image_size, batch_size)
    test_set_x, test_set_y = PickleFile.shared_zeros(image_size, batch_size)

    # ========== build the model ==========
    # start to build the model. prepare the parameters
    print "Building the model..."

    # allocate symbolic variables for the data: 
    # mini batch index, rasterized images, labels
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    # hidden layer
    hidden_layer = HiddenLayer(rng=rng, input=x, 
        n_in=image_size * image_size, n_out=n_hidden, activation=T.tanh)

    # The logistic regression layer gets as input the hidden units
    # of the hidden layer
    log_regression_layer = LogisticRegression(input=hidden_layer.output, 
        n_in=n_hidden, n_out=class_count)

    # L1 norm ; one regularization option is to enforce L1 norm to
    # be small
    l1_norm = abs(hidden_layer.W).sum() + abs(log_regression_layer.W).sum()

    # square of L2 norm ; one regularization option is to enforce
    # square of L2 norm to be small
    l2_sqr = (hidden_layer.W ** 2).sum() + (log_regression_layer.W ** 2).sum()

    # negative log likelihood of the MLP is given by the negative
    # log likelihood of the output of the model, computed in the
    # logistic regression layer
    negative_log_likelihood = log_regression_layer.negative_log_likelihood
    
    # same holds for the function computing the number of errors
    errors = log_regression_layer.errors

    # the parameters of the model are the parameters of the two layer it is
    # made out of
    params = hidden_layer.params + log_regression_layer.params

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = negative_log_likelihood(y)
    cost = cost + l1_reg * classifier.l1_norm + l2_reg * classifier.l2_sqr

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(inputs=[], outputs=errors(y),
        givens={x: test_set_x, y: test_set_y})

    validate_model = theano.function(inputs=[], outputs=errors(y),
        givens={x: valid_set_x, y: valid_set_y})

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[], outputs=cost,
        updates=updates, givens={x: train_set_x, y: train_set_y})

    
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
            print "[%s]" % str(datetime.datetime.now())
            print "training @ iter = %i" % (iter)
            with open(log_file, "a") as log_f:
                log_f.write("[%s]\n" % str(datetime.datetime.now()))
                log_f.write("train @ iter = %i\n" % (iter))

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
                print "[%s]" % str(datetime.datetime.now())
                print ("Epoch %i, batch %i/%i, validation error %f %%" % 
                    (epoch, minibatch_index + 1, n_train_batches, 
                    this_validation_loss * 100.))
                with open(log_file, "a") as log_f:
                    log_f.write("[%s]\n" % str(datetime.datetime.now()))
                    log_f.write(("Epoch %i, batch %i/%i, " 
                        "validation error %f %%\n") % 
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
                    print "[%s]" % str(datetime.datetime.now())
                    print (("    -> test error of best model %f %%") %
                        (test_score * 100.))
                    with open(log_file, "a") as log_f:
                        log_f.write("[%s]\n" % str(datetime.datetime.now()))
                        log_f.write(("    -> test error of best model " 
                            "%f %%\n") % (test_score * 100.))

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
    mlp()
