import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import argparse
import pdb


class NeuralNetwork(object):    
    """
    This is a custom neural netwok package built from scratch with numpy which supports 
    multiclass, multilayer with saving and loading weights.
    It allows training using SGD, inference and plotting of the train loss and test accuracy.
    It's written for educational purposes only.

    The Neural Network as well as its parameters and training method and procedure will 
    reside in this class.

    Arguments:
    size: list of number of neurons per layer. e.g. [784, 512, 256, 10]
    actn_fn: Activation function to choose between Sigmoid, ReLu & Tanh.
    weights: Path of pretrained model weights if any. Default - None
    seed: Seed value to generate random numbers. Default - 42

    Examples
    ---
    >>> nn = NeuralNetwork([784, 512,256, 10])
    >>> history = nn.train(X=X, y=y, batch_size=16, epochs=20, 
                           learning_rate=1e-4, validation_split=0.2)
    
    This means :
    1 input layer with 784 neurons
    1 hidden layer with 512 neurons
    1 hidden layer with 256 neurons
    1 output layer with 10 neuron
    
    """
    def __init__(self, size, actn_fn, weights=None, seed=42):
        """
        Instantiate/Loads the weights and biases of the network.
        weights and biases are attributes of the NeuralNetwork class
        They are updated during the training
        """
        self.seed = seed
        np.random.seed(self.seed)
        self.size = size
        self.actn_fn = actn_fn
        if weights == None:
            self.weights = [np.random.randn(self.size[i], self.size[i-1]) * np.sqrt(2 / self.size[i-1]) for i in range(1, len(self.size))]
            self.biases = [np.random.rand(n, 1) for n in self.size[1:]]
        else:
            self.weights, self.biases = self.load_weights(weights)

    def activation(self, z, derivative=False, type=None):
        """
        Actiavtion Functions
        Three activation functions are defined with derivatives:
        1. Sigmoid 
        2. ReLu & 
        3. Tanh
        Handles both normal and derivative version(pass derivative=True)
        Element wise activation function on the input matrix.

        Arguments:
        z: Pre activation matrix. shape - (num_nodes_layer_l, batch_size)
        derivative - Computes the derivative of the activation function

        Returns:
        Element wise activation function applied matrix. shape - Same as z
        """
        # Sigmoid activation function
        type = self.actn_fn
        if type == "sigmoid":
            if derivative:
                return self.activation(z, type="sigmoid") * (1 - self.activation(z, type="sigmoid"))
            else:
                return 1 / (1 + np.exp(-z))
        
        # ReLu activation function    
        if type == "relu":
            if derivative:
                z[z <= 0.0] = 0.
                z[z > 0.0] = 1.
                return z
            else:
                return np.maximum(0, z)
        
        # Tanh activation function
        if type == "tanh":
            if derivative:
                return self.activation(z, type="tanh") * (1 - self.activation(z, type="tanh"))
            else:
                return np.tanh(z)

    def softmax(self, z):
        """
        Computes the stable softmax of the input matrix.
        Softmax is applied to get class probabilities for each class. Sum of all the class probs is 1.
        Generally softmax is applied to last layer.

        Arguments:
        z: Last layer pre-activated outputs. shape - (num_classes, batch_size)

        Returns:
        Class probabilities for each class. - Shape same as z.

        """
        expz = np.exp(z - np.max(z))
        return expz/expz.sum(axis=0, keepdims=True)

    def cost_function(self, y_true, y_pred):
        """
        Computes Cross Entropy Cost function defined as:
        J = 1/n * sum_i(-y_true_i * log(y_pred_i))

        Arguments:
        y_true: Ground truth values. shape - (1, batch_size)
        y_pred: Predicted values(Output of softmax function). shape - (num_classes, batch_size)

        Returns:
        A scalar value representing cost between y_true and y_pred.
        """
        num_classes = y_pred.shape[0]
        samples = y_pred.shape[1]
        y_true = y_true.reshape(-1,)
        one_hot_labels = np.zeros((samples, num_classes))
        for i in range(samples):  
            one_hot_labels[i, y_true[i]] = 1

        n = y_pred.shape[1]
        y_true = one_hot_labels.T
        return np.mean(np.sum(-y_true * np.log(y_pred+1e-100), axis=0))

    def cost_function_prime(self, y_true, y_pred):
        """
        Computes the dervative of Cost Function with Softmax. The derivative is defiined as:
        delta_J/delta_z_L = y_pred - y_true

        Arguments:
        y_true: Ground truth values. shape - (1, batch_size)
        y_pred: Predicted values(Output of softmax function). shape - (num_classes, batch_size)

        Returns:
        A matrix representing derivative of cost function between y_true and y_pred. shape - same as y_pred
        """
        num_classes = y_pred.shape[0]
        samples = y_pred.shape[1]
        y_true = y_true.reshape(-1,)
        one_hot_labels = np.zeros((samples, num_classes))
        for i in range(samples):  
            one_hot_labels[i, y_true[i]] = 1
        
        y_true = one_hot_labels.T
        cost_prime = y_pred - y_true
        return cost_prime

    def forward(self, input):
        """
        Perform a feed forward computation. Applies softmax as the last layer activation function.

        Arguments:
        input: data to be fed to the network with. shape - (features, batch_size)

        Returns:
        a: ouptut activation (num_classes, batch_size)
        pre_activations: list of pre-activations per layer
        each of shape (n[l], batch_size), where n[l] is the number 
        of neuron at layer l
        activations: list of activations per layer
        each of shape (n[l], batch_size), where n[l] is the number 
        of neuron at layer l
        """
        a = input
        pre_activations = []
        activations = [a]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a  = self.activation(z)
            pre_activations.append(z)
            activations.append(a)
        a = self.softmax(z)
        activations[-1] = a
        return a, pre_activations, activations

    def compute_deltas(self, pre_activations, y_true, y_pred):
        """
        Computes a list containing the values of delta for each layer using 
        a recursion
        Arguments:
        pre_activations: list of of pre-activations. each corresponding to a layer
        y_true: ground truth values of the labels. shape - (1, batch_size)
        y_pred: prediction values of the labels. shape - (num_classes, batch_size)

        Returns:
        deltas: a list of deltas per layer
        """
        delta_L = self.cost_function_prime(y_true, y_pred)# * activation(pre_activations[-1], derivative=True)
        deltas = [0] * (len(self.size) - 1)
        deltas[-1] = delta_L
        for l in range(len(deltas) - 2, -1, -1):
            delta = np.dot(self.weights[l + 1].transpose(), deltas[l + 1]) * self.activation(pre_activations[l],
                                                                                        derivative=True) 
            deltas[l] = delta
        return deltas

    def backpropagate(self, deltas, pre_activations, activations):
        """
        Applies back-propagation and computes the gradient of the loss
        w.r.t the weights and biases of the network

        Arguments:
        deltas: list of deltas computed by compute_deltas
        pre_activations: a list of pre-activations per layer
        activations: a list of activations per layer

        Returns:
        dW: list of gradients w.r.t. the weight matrices of the network
        db: list of gradients w.r.t. the biases (vectors) of the network
        """
        dW = []
        db = []
        deltas = [0] + deltas
        for l in range(1, len(self.size)):
            dW_l = np.dot(deltas[l], activations[l-1].transpose()) 
            db_l = deltas[l]
            dW.append(dW_l)
            db.append(np.expand_dims(db_l.mean(axis=1), 1))
        return dW, db

    def train(self, X, y, batch_size, epochs, learning_rate, validation_split=0.2):
        """
        Trains the network using the gradients computed by back-propagation
        Splits the data in train and validation splits
        Processes the training data by batches and trains the network using batch gradient descent.
        Saves weights of the best accuracy among all epochs.

        Arguments:
        X: input data. shape - (features, samples)
        y: input labels. shape - (1, samples)
        batch_size: number of data points to process in each batch
        epochs: number of epochs for the training
        learning_rate: value of the learning rate
        validation_split: percentage of the data for validation
    
        Returns:
        history: dictionary of train and validation metrics per epoch
            train_loss: training loss
            test_acc: validation accuracy

        This history is used to plot the performance of the model
        """
        history_train_losses = []
        history_test_accuracies = []

        x_train, x_test, y_train, y_test = train_test_split(X.T, y.T, test_size=validation_split, )
        x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T 

        epoch_iterator = range(epochs)
        best_weights_acc = 0

        for e in epoch_iterator:
            if x_train.shape[1] % batch_size == 0:
                n_batches = int(x_train.shape[1] / batch_size)
            else:
                n_batches = int(x_train.shape[1] / batch_size ) - 1

            x_train, y_train = shuffle(x_train.T, y_train.T)
            x_train, y_train = x_train.T, y_train.T

            batches_x = [x_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]
            batches_y = [y_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]

            train_losses = []
            test_accuracies = []

            dw_per_epoch = [np.zeros(w.shape) for w in self.weights]
            db_per_epoch = [np.zeros(b.shape) for b in self.biases] 
            print("Epoch {}/{}".format(e+1, epochs))
            
            for idx, (batch_x, batch_y) in enumerate(zip(batches_x, batches_y)):
                batch_y_pred, pre_activations, activations = self.forward(batch_x)
                deltas = self.compute_deltas(pre_activations, batch_y, batch_y_pred)
                dW, db = self.backpropagate(deltas, pre_activations, activations)
                for i, (dw_i, db_i) in enumerate(zip(dW, db)):
                    dw_per_epoch[i] += dw_i / batch_size
                    db_per_epoch[i] += db_i / batch_size

                train_loss = self.cost_function(batch_y, batch_y_pred)
                train_losses.append(train_loss)

                print("{}/{}".format(idx, n_batches), end='\r')

            # weight update
            for i, (dw_epoch, db_epoch) in enumerate(zip(dw_per_epoch, db_per_epoch)):
                self.weights[i] = self.weights[i] - learning_rate * dw_epoch
                self.biases[i] = self.biases[i] - learning_rate * db_epoch

            print("Training Loss:{}".format(sum(train_losses)/len(train_losses)))
            y_test_pred = self.predict(x_test)
            test_accuracy = accuracy_score(y_test.T, y_test_pred.T)
            test_accuracies.append(test_accuracy)
            
            print("Testing Accuracy:{}".format(test_accuracy))

            if test_accuracy > best_weights_acc:
                print("Saving weights")
                best_weights_acc = test_accuracy
                self.save_weights() 

            print()
            history_train_losses.append(np.mean(train_losses))
            history_test_accuracies.append(np.mean(test_accuracies))

        history = {'train_loss': history_train_losses, 
                   'test_acc': history_test_accuracies}
        return history

    def predict(self, a):
        """
        Computes prediction of the validation/test data by doing a forward pass.
        Softmax is applied to the last layer to compute class probs
        and then maximum valued index is taken.

        Arguments:
        a: Input data. shape - (features, samples)

        Returns:
        Predictions, shape - (1, samples)
        """
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.activation(z)
        a = self.softmax(z)
        predictions = np.argmax(a, axis=0).reshape(1, -1).astype(int)
        return predictions

    def save_weights(self):
        """
        Saves the model architecture and weights of each layers as dictionary.

        Arguments: None
        Returns: None
        """
        model = {"size":self.size}
        for i in range(1, len(self.size)):
            model["W"+str(i)] = self.weights[i-1]
            model["b"+str(i)] = self.biases[i-1]
        np.save('best_acc_weights.npy', model)
        return None

    def load_weights(self, weights):
        """
        Loads the saved model if the model architecture matches.

        Arguments:
        weights: Path to the saved model using save_weights method.

        Returns:
        saved weights and biases of each layers.
        """
        model = np.load(weights).item()
        size = model["size"]

        if size == self.size:
            weights = [model["W"+str(i)] for i in range(1, len(size))]
            biases = [model["b"+str(i)] for i in range(1, len(size))]
        else:
            raise ValueError("Model dimension doesn't matches")
        return weights, biases


def load_data(train_path, test_path):
    """
    Loads the train and test dataset from the given path.
    Fashion-MNIST dataset is used.

    Arguments:
    train_path: Path for the train dataset.
    test_path: Path for the test dataset.

    Returns:
    X: X_train. shape - (samples, features) e.g. (60000, 784)
    y: y_train. shape - (samples,) e.g. (60000,)
    X_test: X_test. shape - (samples, features) e.g. (9674, 784)
    """
    data = pd.read_csv(train_path, delimiter=',')
    data = data.values
    X = data[:, 1:]
    y = data[:, 0]

    test_data = pd.read_csv(test_path, delimiter=',')
    test_data = test_data.values
    X_test = test_data[:]

    return X, y, X_test

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--activation_fn", type=str, default="relu", help="Activation Function")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--hidden_layers", type=str, default='100', help="hideen layers size")
    parser.add_argument("--weights_path", type=str, default=None, help="Path to saved weights")    
    opt = parser.parse_args()
    print(opt)

    # Loading Data
    train_path = './Apparel/apparel-trainval.csv'
    test_path = './Apparel/apparel-test.csv'
    X, y, X_test = load_data(train_path, test_path)

    # Data Normalization
    X = X/255.
    X_test = X_test/255.

    X, y = X.T, y.reshape(1, -1)
    X_test = X_test.T
  
    print("Data Loaded")
    print("Starting")

    # Hyperparameters
    num_classes = opt.num_classes
    batch_size = opt.batch_size
    activation_fn = opt.activation_fn
    epochs = opt.epochs
    lr = opt.lr
    hidden_layers = [int(i) for i in opt.hidden_layers.split(',')]
    size = [X.shape[0]] + hidden_layers + [num_classes]

    # Training
    neural_net = NeuralNetwork(size=size, actn_fn=activation_fn, seed=42)
    history = neural_net.train(X=X, y=y, batch_size=batch_size, epochs=epochs, 
                               learning_rate=lr, validation_split=0.2)    
    history_train_losses = history["train_loss"]
    history_test_accuracies = history["test_acc"]

    # Train Loss and Test accuracy plot
    plt.subplot(1,2,1)
    plt.plot(list(range(1, epochs+1)), history_train_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Train Loss")

    plt.subplot(1,2,2)
    plt.plot(list(range(1, epochs+1)), history_test_accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.show()

    # Predictions
    predictions = neural_net.predict(X_test)