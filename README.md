# NueralNetworkClassifier

####Backpropagation-trained stochastic gradient descent neural network classifier in Java

to run on command line use 

./NeuralNetworkClassifier -task c|r|l [-batch mb] L ss data_cfg_fn

* Task flag : c uses classification mode, r uses regression mode, l uses logistic mode
* Batch flag : uses batch mode for mb=0 and minibatch mode for mb>1
* L : number of hidden layer nodes
* ss : step size (learning rate) (floating point)
* data_cfg_fn : name of the data config file

**DATA CONFIG FILE FORMAT**

key/value pairs seperated by newlines in a txt file with the following keys
* N_TRAIN         positive integer  : number of datapoints in the training set
* N_DEV           positive integer  : number of datapoints in the dev set
* TRAIN_X_FN      string            : relative or absolute path of the training set feature file
* TRAIN_T_FN      string            : relative or absolute path of the training set target file
* DEV_X_FN        string            : relative or absolute path of the dev set feature file
* DEV_T_FN        string            : relative or absolute path of the dev set training file
* D               positive integer  : dimension of input data
* C               positive integer  : number of classes
