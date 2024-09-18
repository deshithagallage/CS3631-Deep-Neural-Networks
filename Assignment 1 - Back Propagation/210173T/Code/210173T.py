import numpy as np
import csv

def read_csv(file_name):
    list =[]
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader: 
            list.append(row)
    f.close()
    return list

def write_csv(file_name, data):
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    f.close()

# Relu activation function
def relu(x):
    return np.maximum(0, x)

# Relu derivative
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Softmax activation function
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

# Cross entropy loss function
def cross_entropy_loss(predicted, actual):
    return -np.sum(actual * np.log(predicted))  # make sure to avoid log(0)

def get_parameters(weights, biases, L, layers_size):
    parameters = {}
    rowcounter = 0  # for navigating through the rows in weights array which has weights of all layers together
    
    # store these weights layer wise into parameters dictionary
    for l in range(L):
        parameters["W" + str(l + 1)] = []
        for i in range(layers_size[l]):
            weightsub = np.asarray(weights[rowcounter], dtype=np.float32)
            parameters["W" + str(l + 1)].append(weightsub)
            rowcounter += 1
        parameters["W" + str(l + 1)] = np.array(parameters["W" + str(l + 1)])

    # store the biases layer wise into parameters dictionary
    for l in range(L):
        parameters["b" + str(l + 1)] = list(map(float, biases[l]))

    parameters["b" + str(l + 1)] = np.array(parameters["b" + str(l + 1)], dtype=np.float32)

    return parameters

def forward_propagation(X, L, parameters):
    # X: input layer
    # L: number of layers
    # parameters: dictionary containing weights and biases

    A = X.T  # Transpose X to match dimensions
    caches = {}
    for l in range(1, L):
        W = parameters['W' + str(l)].T
        b = parameters['b' + str(l)]
        Z = np.dot(W, A) + b
        A = relu(Z)

        caches["A" + str(l)] = A
        caches["W" + str(l)] = W
        caches["Z" + str(l)] = Z

    # Output layer
    W = parameters['W' + str(L)].T
    b = parameters['b' + str(L)]
    Z = np.dot(W, A) + b
    A = softmax(Z)

    caches["A" + str(L)] = A
    caches["W" + str(L)] = W
    caches["Z" + str(L)] = Z

    return A, caches

def backward_propagation(X, Y, L, caches, precision=16):
    # X: input layer
    # Y: actual output
    # L: number of layers
    # caches: dictionary containing A, W, Z of each layer

    gradients = {}
    Y = np.asarray(Y, dtype=np.float32)  # Convert Y to np array (if not already)

    caches["A0"] = X  # Set the input layer in the caches
    dZ = caches['A' + str(L)] - Y  # Compute error at the output layer (A - Y)

    # Compute dW for the output layer (L-th layer)
    dW = np.outer(caches['A' + str(L - 1)], dZ)
    dW = np.round(dW, precision)

    # Store gradients for the output layer
    gradients["dW" + str(L)] = dW
    gradients["db" + str(L)] = dZ

    # Initialize dAPrev for backpropagation through hidden layers
    dAPrev = (caches["W" + str(L)].T).dot(dZ.T)

    # Loop over hidden layers (L-1, L-2, ..., 1)
    for l in range(L - 1, 0, -1):
        # Compute dZ for the current layer based on ReLU derivative (ReLU's derivative is 1 for positive Z, 0 otherwise)
        dZ = np.where(caches['Z' + str(l)] > 0, dAPrev, 0)

        # Compute dW for the current layer
        dW = np.outer(dZ, caches['A' + str(l - 1)])

        if l > 1:
            # Compute dAPrev for the next layer back
            dAPrev = (caches["W" + str(l)].T).dot(dZ)

        # Round the gradients to the specified precision
        dW = np.round(dW, precision)

        # Store the gradients for the current layer
        gradients["dW" + str(l)] = dW.T  # Transpose dW to match dimensions
        gradients["db" + str(l)] = dZ

    return gradients

if __name__ == "__main__":

    NN = "100-40-4"     # Neural network architecture
    INPUT_SIZE = 14     # Input size

    WEIGHTS = f'Task_1/b/w-{NN}.csv' # Weights file
    BIASES = f'Task_1/b/b-{NN}.csv'  # Biases file

    L = len(NN.split("-"))  # Number of layers
    layers = [INPUT_SIZE] + [int(x) for x in NN.split("-")]

    X = [-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]
    X = np.asarray(X, dtype=int)
    Y = [0, 0, 0, 1]

    weights = read_csv(WEIGHTS)
    biases = read_csv(BIASES)

    filtered_weights = [
        [float(value) for value in row[1:] if value.strip() != ''] for row in weights
    ]
    filtered_biases = [
        [float(value) for value in row[1:] if value.strip() != ''] for row in biases
    ]

    parameters = get_parameters(filtered_weights, filtered_biases, L, layers)

    A, caches = forward_propagation(X, L, parameters)
    gradients = backward_propagation(X, Y, L, caches)

    # Write the gradients to a CSV file
    for i in range(L):
        with open("dw.csv", 'a+', newline='') as f:
            write = csv.writer(f)
            write.writerows(gradients[f"dW{i + 1}"])
        f.close()

    # Write the biases to a CSV file
    db_data = [gradients[f"db{i + 1}"] for i in range(L)]
    write_csv("db.csv", db_data)    
