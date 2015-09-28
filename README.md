# NueralNetworkClassifier

####Backpropagation-trained stochastic gradient descent neural network classifier in Java

to run on command line use 

./NeuralNetworkClassifier -task c|r|l [-batch mb] L ss data_cfg_fn

* Task flag : c uses classification mode, r uses regression mode, l uses logistic mode
* Batch flag : uses batch mode for mb=0 and minibatch mode for mb>1
* L : number of hidden layer nodes
* ss : step size (learning rate) (floating point)
* data_cfg_fn : name of the data config file

**Data config file format**

key/value pairs seperated by newlines in a txt file with the following keys
* N_TRAIN         positive integer  : number of datapoints in the training set
* N_DEV           positive integer  : number of datapoints in the dev set
* TRAIN_X_FN      string            : relative or absolute path of the training set feature file
* TRAIN_T_FN      string            : relative or absolute path of the training set target file
* DEV_X_FN        string            : relative or absolute path of the dev set feature file
* DEV_T_FN        string            : relative or absolute path of the dev set training file
* D               positive integer  : dimension of input data
* C               positive integer  : number of classes


**Feature File Format**

* an input feature file is N lines long
* each line consists of D floating point values, delimited by spaces (D is the dimension of the data)
* no assumption is made about the number of decimal places

Example : N = 3, D = 4

1.2 3.0 6.6 2.3

4.5 7.1 1.4 9.8

6.7 2.2 1.1 3.4


**Target File Format**

* a target file is N lines long
* these C values define the target vector for the datapoint
* each line has C floating point values, delimited by spaces (C is the output dimension)
* no assumption is made about the number of decimal places
* Can be either a *one-hot vector* (using classification) or a *floating point* (using regression) 

Example : N = 3, C =3

0 1 0

1 0 0

1 0 0

*OR*

0.93 0.57 0.01

0.23 0.00 0.66

0.00 1.00 0.00
