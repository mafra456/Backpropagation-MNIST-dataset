import numpy as np

#Sigmoid Function for non-linearity
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def gradient(a):
    return (a *(1-a))

def gradient_descent(m, X, y,_y, w1, w2,batch_size):
    num_iter = 100
    for i in xrange(num_iter):
        w1, w2 = backpropagation(m, X, y,_y, w1, w2, batch_size)

def stochastic_gradient_descent(m, X, y,_y, w1, w2, batch_size):
    num_iter = 70
    for i in xrange(num_iter):
        w1, w2 = backpropagation(m, X, y,_y, w1, w2, batch_size)
def mini_batch(m, X, y,_y, w1, w2, batch_size):
    num_iter = 100
    for i in xrange(num_iter):
        w1, w2 = backpropagation(m, X, y,_y, w1, w2, batch_size)

def forward_pass(w1, w2, X):
    m , n = X.shape
    layer2 = sigmoid(np.dot(X, w1.T))
    layer2 = np.append(np.ones((m,1)), layer2, axis = 1)
    layer3 = sigmoid(np.dot(layer2, w2.T))
    return np.argmax(layer3, axis = 1)

def accuracy(w1, w2, X, _y):
    m, n = X.shape
    fp = forward_pass(w1,w2,X)
    fp = fp.reshape(m,1)
    accuracy = 100 * np.mean(fp == _y)
    print(100 - accuracy)

    #Used to read the matrices of weights from a txt in order to compare results
def read_weights(size):
    w1 = np.genfromtxt('weights/weight1_'+str(size)+'.txt', delimiter=' ')
    w2 = np.genfromtxt('weights/weight2_'+str(size)+'.txt', delimiter=' ')
    return w1,w2

def initialize_weights(i, j):
    eps = 0.12
    w = np.random.uniform(-eps, eps, (i,j+1))
    return w

def main():
    #Matrix of Inputs
    X = np.genfromtxt('data_tp1.txt', delimiter=',')
    _y = np.array(X[:, 0])
    _y = _y.reshape(5000,1)
    X[:, 0] = 1

    #Matrix of Real Outputs
    y = np.zeros((5000,10))
    for i in xrange(_y.size):
        y[i][int(_y[i])] = 1

    m,n = X.shape
    num_hidden_layers = 25

    #Matrices of Weights
    w1 = initialize_weights(26,784)
    w2 = initialize_weights(10,26)

    #Reads weights from a file when executing for testing
    #w1,w2 = read_weights(num_hidden_layers)

    gradient_descent(m, X, y,_y, w1, w2, m)
    #stochastic_gradient_descent(m, X, y,_y, w1, w2, 1)
    #mini_batch(m, X, y,_y, w1, w2, 50)


def backpropagation(m, X, y, _y, w1, w2,batch_size):
    #learning rate
    r = 0.5
    w1_err = np.zeros(w1.shape) 
    w2_err = np.zeros(w2.shape)
    cost = 0
    for i in xrange(m/batch_size):
        for j in xrange(batch_size):
            #Feed Forward
            layer1 = X[batch_size*i + j, :]
            layer2 = sigmoid(np.dot(w1, layer1))
            layer2 = np.append(np.ones((1,1)), layer2)
            layer3 = sigmoid(np.dot(layer2, w2.T))

            #Calculates the loss function
            #cost += np.sum(-y[i, :] * np.log(layer3) - (1.0 -y[i, :]) * np.log(1.0 - layer3))

            #Calculates errors of the layers
            layer3_err = y[batch_size*i + j, :] - layer3
            w2_err += np.outer(layer3_err, layer2)

            layer2_err = layer3_err.dot(w2)
            layer2_delta = layer2_err * gradient(layer2)
            w1_err += np.outer(layer2_delta[1:], layer1)

        # Takes average of error accumulated
        w1_err = w1_err/m
        w2_err = w2_err/m

        #Takes average of cost
        #print(cost/m)

        w1 = w1 + w1_err*r
        w2 = w2 + w2_err*r

        #para o mini_batch
        w1_err = 0
        w2_err = 0

    accuracy(w1, w2, X, _y)

    return w1, w2

if __name__ == '__main__':
    main()