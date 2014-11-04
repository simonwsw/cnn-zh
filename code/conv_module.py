import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

# class logistic regression
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out), 
            dtype=theano.config.floatX), name='W', borrow=True)
        
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,), 
            dtype=theano.config.floatX), name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    # return the mean of the negative log-likelihood of the prediction
    # of this model under a given target distribution.
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    # return a float representing the number of errors in the minibatch 
    # over the total number of examples of the minibatch
    # zero one loss over the size of the minibatch
    def errors(self, y):

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', 
                ('y', target.type, 'y_pred', self.y_pred.type))

        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

# class hidden layer
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
            activation=T.tanh):
        
        self.input = input
        
        # initialize W if not given
        if W is None:
            
            # initialize with given fan in and fan out
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            
            # if activation function is sigmoid, W should time 4
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        # initialize b if not given, with zeros
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # output the perceptron output value
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
            else activation(lin_output))
        
        # parameters of the model
        self.params = [self.W, self.b]

# convolution neural network convolute and pooling layer parameter
class LeNetConvPoolParam(object):
    def __init__(self, input_image_size, input_feature_num, 
            filter_size, pooling_size, kernel):
        self.input_image_size = input_image_size
        self.input_feature_num = input_feature_num
        self.filter_size = filter_size
        self.pooling_size = pooling_size
        self.kernel = kernel
        
        # calculate output size
        self.output_size = ((input_image_size - filter_size + 1) / 
            pooling_size)

# convolution neural network convolute and pooling layer
class LeNetConvPoolLayer(object):
    
    # filter shape: 
    # mini-batch size, num input feature maps, image height, image width
    # image shape: 
    # num feature maps at layer m, num feature maps at layer m-1, 
    # filter height, filter width
    def __init__(self, rng, input, filter_shape, image_shape, 
            poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # fan in to each hidden unit:
        # num input feature maps * filter height * filter width
        fan_in = numpy.prod(filter_shape[1:])

        # fan out when ower layer receives a gradient from:
        # num output feature maps * filter height * filter width / 
        # pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / 
            numpy.prod(poolsize))
        
        # initialize weights with random weights
        W_bound = numpy.sqrt(6.0 / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), 
            dtype=theano.config.floatX), borrow=True)

        # the bias is a 1D tensor: num output feature map * 1
        # one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W, 
            filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
            ds=poolsize, ignore_border=True)

        # add the bias term. reshape it to a tensor of shape 
        # (1, n_filters, 1, 1). bias will be broadcasted across mini-batches 
        # and feature map width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]