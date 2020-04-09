import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Sigmoid function.
    This function accepts any shape of np.ndarray object as input and perform sigmoid operation.
    """
    return 1 / (1 + np.exp(-x))


def der_sigmoid(y):
    """ First derivative of Sigmoid function.
    The input to this function should be the value that output from sigmoid function.
    """
    return y * (1 - y)


class GenData:
    @staticmethod
    def _gen_linear(n=100):
        """ Data generation (Linear)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data = np.random.uniform(0, 1, (n, 2))

        inputs = []
        labels = []

        for point in data:
            inputs.append([point[0], point[1]])

            if point[0] > point[1]:
                labels.append(0)
            else:
                labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def _gen_xor(n=100):
        """ Data generation (XOR)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data_x = np.linspace(0, 1, n // 2)
        inputs = []
        labels = []

        for x in data_x:
            inputs.append([x, x])
            labels.append(0)

            if x == 1 - x:
                tmp = 0.5
                while tmp is 0.5:
                    tmp = np.random.rand()
                print (tmp)
                # if tmp < 0.5:
                    # inputs.append([tmp, tmp])
                    # labels.append(0)
                # else:
                inputs.append([tmp, 1 - tmp])
                labels.append(1)

                continue

            inputs.append([x, 1 - x])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def fetch_data(mode, n):
        """ Data gather interface

        Args:
            mode (str): 'Linear' or 'XOR', indicate which generator is used.
            n (int):    the number of data points generated in total.
        """
        assert mode == 'Linear' or mode == 'XOR'

        data_gen_func = {
            'Linear': GenData._gen_linear,
            'XOR': GenData._gen_xor
        }[mode]

        return data_gen_func(n)


class SimpleNet:
    def __init__(self, hidden_size, num_step=2000, print_interval=100, lr=0.1):
        """ A hand-crafted implementation of simple network.

        Args:
            hidden_size:    the number of hidden neurons used in this model.
            num_step (optional):    the total number of training steps.
            print_interval (optional):  the number of steps between each reported number.
        """
        self.num_step = num_step
        self.print_interval = print_interval
        self.lr = lr

        # Model parameters initialization
        # Please initiate your network parameters here.

        # W0, W1, W2 corresponding to input layer, hidden layer1 , hidden layer2
        self.weights = [np.random.rand(2, hidden_size), np.random.rand(hidden_size, 3), np.random.rand(3, 1)]


    @staticmethod
    def plot_result(data, gt_y, pred_y):
        """ Data visualization with ground truth and predicted data comparison. There are two plots
        for them and each of them use different colors to differentiate the data with different labels.

        Args:
            data:   the input data
            gt_y:   ground truth to the data
            pred_y: predicted results to the data
        """
        assert data.shape[0] == gt_y.shape[0]
        assert data.shape[0] == pred_y.shape[0]

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title('Ground Truth', fontsize=18)

        for idx in range(data.shape[0]):
            if gt_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Prediction', fontsize=18)

        for idx in range(data.shape[0]):
            if pred_y[idx] < 0.5:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.show()

    def forward(self, inputs):
        """ Implementation of the forward pass.
        It should accepts the inputs and passing them through the network and return results.
        """

        # Order: sig(sig(sig(input*W0)*W1)*W2)
        out_input = np.dot(inputs, self.weights[0])
        out_input = sigmoid(out_input)
        # print (out_input)

        out_hd1 = np.dot(out_input, self.weights[1])
        out_hd1 = sigmoid(out_hd1)
        # print (out_hd1)

        y = np.dot(out_hd1, self.weights[2])
        y = sigmoid(y)
        # print (y)

        self.forward_results = [out_input, out_hd1, y]
        

        return y

    def backward(self, X):
        """ Implementation of the backward pass.
        It should utilize the saved loss to compute gradients and update the network all the way to the front.
        """
        # reference: https://towardsdatascience.com/back-propagation-the-easy-way-part-1-6a8cde653f65
        # 推導過程: https://hackmd.io/GhEXIQkvRSKiShFzh71nXg?view

        # Gradient descent for W3
        dLdA3 = self.error # d(y - y_hat) / d(y)
        dA3dZ3 = der_sigmoid(self.forward_results[2])
        dZ3dW3 = self.forward_results[1]
        
        # Gradient descent for W2
        dZ3dA2 = self.weights[2].T
        dA2dZ2 = der_sigmoid(self.forward_results[1])
        dZ2dW2 = self.forward_results[0]

        # Gradient descent for W1
        dZ2dA1 = self.weights[1].T
        dA1dZ1 = der_sigmoid(self.forward_results[0])
        dZ1dW1 = X

        self.weights[2] = self.weights[2] - self.lr * (dLdA3 * dA3dZ3 * dZ3dW3).T
        self.weights[1] = self.weights[1] - self.lr * (dLdA3 * dA3dZ3 * (dZ3dA2 * dA2dZ2).T * dZ2dW2).T
        self.weights[0] = self.weights[0] - self.lr * (dLdA3 * dA3dZ3 * np.dot(dZ3dA2 * dA2dZ2, dZ2dA1 * dA1dZ1).T * X).T

        return


    def train(self, inputs, labels):
        """ The training routine that runs and update the model.

        Args:
            inputs: the training (and testing) data used in the model.
            labels: the ground truth of correspond to input data.
        """
        # make sure that the amount of data and label is match
        assert inputs.shape[0] == labels.shape[0]

        n = inputs.shape[0]

        for epochs in range(self.num_step):
            # if epochs % self.print_interval*10 == 0:
            #     self.lr *= 0.9
            for idx in range(n):
                # operation in each training step:
                #   1. forward passing
                #   2. compute loss
                #   3. propagate gradient backward to the front
                self.output = self.forward(inputs[idx:idx+1, :])
                self.error = self.output - labels[idx:idx+1, :]
                self.backward(inputs[idx:idx+1, :])

            if epochs % self.print_interval == 0:
                print('Epochs {}: '.format(epochs))
                self.test(inputs, labels)

        print('Training finished')
        self.test(inputs, labels)

    def test(self, inputs, labels):
        """ The testing routine that run forward pass and report the accuracy.

        Args:
            inputs: the testing data. One or several data samples are both okay.
                The shape is expected to be [BatchSize, 2].
            labels: the ground truth correspond to the inputs.
        """
        n = inputs.shape[0]

        error = 0.0
        for idx in range(n):
            result = self.forward(inputs[idx:idx+1, :])
            error += abs(result - labels[idx:idx+1, :])

        error /= n
        print('accuracy: %.2f' % ((1 - error)*100) + '%', 'cost:', self.error[0][0])
        print('')


if __name__ == '__main__':
    data, label = GenData.fetch_data('Linear', 70)
    lr = 0.5
    
    net = SimpleNet(6, num_step=8000, lr=lr)
    net.train(data, label)

    pred_result = np.round(net.forward(data))
    SimpleNet.plot_result(data, label, pred_result)
