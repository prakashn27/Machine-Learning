import pickle
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  1.0 / (1.0 + np.exp(-1.0 * z))
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])  
    initialDataMatrix = np.array([])
    labelVector = np.array([])
    
    #Train & Validation data -
    for i in range(10):
        trainx = mat.get('train'+str(i))
        initialDataMatrix = np.vstack((initialDataMatrix,trainx)) if initialDataMatrix.size else trainx
        for j in range(len(trainx)):
            labelVector = np.append(labelVector, str(i))
    
    #Test data - 
    for i in range(10):
        testx = mat.get('test'+str(i))
        test_data = np.vstack((test_data,testx)) if test_data.size else testx
        for j in range(len(testx)):
            test_label = np.append(test_label, str(i))
            
    #Split the matrix into Training and Validation Data
    initialRows = range(initialDataMatrix.shape[0])
    aperm = np.random.permutation(initialRows)
    Training_Data = initialDataMatrix[aperm[0:50000],:]
    validation_data = initialDataMatrix[aperm[50000:],:]
    
    #Split the label vector into training label vector and validation label vector
    train_label = labelVector[aperm[0:50000]]
    validation_label = labelVector[aperm[50000:]]
    
    #Convert the data into double
    Training_Data = np.double(Training_Data)
    test_data = np.double(test_data)
    validation_data = np.double(validation_data)
    
    #Normalize the training and the test matrices 
    Training_Data = Training_Data / 255
    test_data   = test_data / 255
    validation_data = validation_data / 255
    
    #Feature Selection
    columnIndices = np.array([])
    for i in range(0,Training_Data.shape[1]):
        col = Training_Data[:,i]
        if np.all(col == col[0]):
            columnIndices = np.append(columnIndices,i)
    
    train_data = np.delete(Training_Data, columnIndices, 1)
    val_data = np.delete(validation_data, columnIndices, 1)
    testing_data = np.delete(test_data, columnIndices, 1)
    
    return train_data, train_label, val_data, validation_label, testing_data, test_label    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, the training data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval, target_matrix = args
        
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    (rows, features) = training_data.shape
    errorFunctionValue = 0
    cumulativeDerivativeOutput = [[0 for y in range(n_hidden + 1)] for x in range(n_class)]
    cumulativeDerivativeHidden = [[0 for y in range(n_input + 1)] for x in range(n_hidden)]
    
    ##changing into ndArray from list
    cumulativeDerivativeOutput = np.array(cumulativeDerivativeOutput)
    cumulativeDerivativeHidden = np.array(cumulativeDerivativeHidden)
    
    ##Adding bias node in the Input Layer
    bias_hidden = np.ones((training_data.shape[0], 1))
    training_data = np.hstack((training_data, bias_hidden))
    
    a = np.dot(training_data, np.transpose(w1))
    z = sigmoid(a)
    
    bias_output = np.ones((z.shape[0], 1))
    z = np.hstack((z, bias_output))
    
    b = np.dot(z, np.transpose(w2))
    o = sigmoid(b)
    
    ##delta value calculation   
    target_matrix = target_matrix.reshape((rows, n_class))
    dell = np.subtract(o, target_matrix)
    
    # Construct the derivative of error function with respect to the weight from the 
    # hidden unit j to output unit l 
    # Matrix : l x j
    cumulativeDerivativeOutput = np.dot(dell.T, z)
    
    # Construct the derivative of error function with respect to the weight from the 
    # input unit p to hidden unit j 
    # Matrix : j x p
    summation = np.dot(dell, w2)     
    
    one = np.ones(z.shape)
    oneMinusZ = np.subtract(one, z)
    productZ = np.multiply(oneMinusZ,z)
    temp = np.multiply(productZ, summation)
    cumulativeDerivativeHidden = np.dot(temp.T,training_data)
    cumulativeDerivativeHidden = np.delete(cumulativeDerivativeHidden, cumulativeDerivativeHidden.shape[0] - 1, 0)
    
    #Calculate the negative log likelihood error function
    ones = np.ones((target_matrix.shape))
    oneMinusTarget = ones - target_matrix 
    oneMinusO = ones - o
    first_term = np.multiply(target_matrix, np.log(o))
    second_term = np.multiply(oneMinusTarget, np.log(oneMinusO))
    errorFunctionValue = np.sum(np.add(first_term, second_term))
    errorFunctionValue = (-1)*(errorFunctionValue/rows)
    
    cumulativeDerivativeOutput = cumulativeDerivativeOutput + np.multiply(lambdaval, w2)
    cumulativeDerivativeHidden = cumulativeDerivativeHidden + np.multiply(lambdaval, w1)
    
    cumulativeDerivativeOutput = cumulativeDerivativeOutput/rows
    cumulativeDerivativeHidden = cumulativeDerivativeHidden/rows
    
    firstLayerSummation = np.sum(np.multiply(w1, w1))
    secondLayerSummation = np.sum(np.multiply(w2, w2))
    
    obj_val = errorFunctionValue + ((lambdaval/(2*rows))*(firstLayerSummation + secondLayerSummation))
    obj_grad = np.array([])
    obj_grad = np.concatenate((cumulativeDerivativeHidden.flatten(), cumulativeDerivativeOutput.flatten()),0)

    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    ##Adding bias node in the Input Layer
    labels = np.array([])
    bias_hidden = np.ones((data.shape[0], 1))
    training_data = np.hstack((data, bias_hidden))

    a = np.dot(training_data, np.transpose(w1))
    z = sigmoid(a)

    bias_output = np.ones((z.shape[0], 1))
    z = np.hstack((z, bias_output))

    b = np.dot(z, np.transpose(w2))
    o = sigmoid(b)

    for num in range(o.shape[0]):
        row = o[num,:]
        labels = np.append(labels,np.argmax(row))
    return labels
    



"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 100;
	   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.6;

#Construct the Target Matrix for all input data
encodingMatrix = np.identity(n_class)
target_matrix = np.array([])
for digit in train_label:   #get the training label and append it to the encoding matrix
    correspondingVector = encodingMatrix[np.double(digit), :]
    target_matrix = np.concatenate((target_matrix, correspondingVector), 0)

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval, target_matrix)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == np.double(train_label)).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == np.double(validation_label)).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Testing Dataset
print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == np.double(test_label)).astype(float))) + '%')

t = (n_hidden, w1, w2, lambdaval)
with open('params.pickle', 'wb') as params:
	p = pickle.Pickler(params)
	p.dump(t)