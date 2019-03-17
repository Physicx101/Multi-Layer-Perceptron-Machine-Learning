import pandas as pd
import numpy as np
from random import seed
from random import random
import matplotlib.pyplot as plt

# memuat data dan menambah kolom untuk representasi biner
filename = 'iris.csv'
dataset = pd.read_csv(filename)
dataset.columns = ['x1', 'x2', 'x3', 'x4', 'species']
dataset['species'] = dataset.species.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0.0, 1.0, 2.0])
rd_data = dataset.values.tolist()
np.random.shuffle(rd_data)


# fungsi untuk menghitung delta
def delta(output):
    return output * (1.0 - output)


# Fungsi aktivasi sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Fungsi untuk menghitung hasil per neuron
def calculate(weight, theta):
    result = weight[-1]
    for i in range(len(weight) - 1):
        result += weight[i] * theta[i]
        return result


# Fungsi error (loss function)
def error(actual, predicted):
    return np.sum(0.5 * (predicted - actual ** 2))


# Membuat prediksi jika aktivasi >= 0.5 maka = 1
def predict(activation):
    return 1 if activation >= 0.5 else 0


# Menginisialisasi weight dari network
def initialize(n_input, n_hidden, n_output):
    net = list()
    hidden_layer = [{'weights': [random() for i in range(n_input + 1)]} for i in range(n_hidden)]
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_output)]
    net.append(hidden_layer)
    net.append(output_layer)
    return net


# Fungsi untuk feed forward dari input ke output
def feed_forward(net, row):
    inputs = row
    for layer in net:
        new_input = []
        for n in layer:
            result = calculate(n['weights'], inputs)
            n['out'] = sigmoid(result)
            new_input.append(n['out'])
        inputs = new_input
    return inputs


# Fungsi untuk back propagation
def back_propagation(network, target):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                e = 0.0
                for n in network[i + 1]:
                    e += (n['weights'][j] * n['delta'])
                errors.append(e)
        else:
            for j in range(len(layer)):
                n = layer[j]
                errors.append(target[j] - n['out'])
        for j in range(len(layer)):
            n = layer[j]
            n['delta'] = errors[j] * delta(n['out'])


# Fungsi untuk update weight
def update_weight(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['out'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learning_rate * neuron['delta']


# Fungsi untuk training & validasi netwrok

def train_data(network, train, val, learning_rate, epoch, n_out):
    for ep in range(epoch):
        correct = 0
        sum_error = 0
        for row in train:
            outputs = feed_forward(network, row)
            expected = [0 for i in range(n_out)]
            expected[int(row[-1])] = 1
            sum_error += sum([error(expected[i], (outputs[i])) for i in range(len(expected))])
            back_propagation(network, expected)
            update_weight(network, row, learning_rate)

            # membuat prediksi dari hasil aktivasi
            for i in range(len(expected)):
                outputs[i] = predict(outputs[i])
            if expected == outputs:
                correct += 1
        validate_data(network, val, n_out)
        error_train.append(sum_error / len(train))
        acc_train.append(correct / len(train))


# Fungsi untuk validasi network
def validate_data(network, val, n_out):
    score = 0
    sum_error = 0

    for row in val:
        outputs = feed_forward(network, row)
        expected = [0 for i in range(n_out)]
        expected[int(row[-1])] = 1
        sum_error += sum([error(expected[i], (outputs[i])) for i in range(len(expected))])

        # membuat prediksi dari hasil aktivasi
        for i in range(len(expected)):
            outputs[i] = predict(outputs[i])

        if expected == outputs:
            score += 1

    error_val.append(sum_error / len(train))
    acc_val.append(score / len(train))


# 70% training, 30% validasi
total = len(rd_data)
train_length = int(0.7 * total)
val_length = int(0.3 * total)

train = rd_data[:train_length]
val = rd_data[val_length:]

# Training backprop algorithm
error_train = []
acc_train = []
error_val = []
acc_val = []
seed(8080)

input_layer = 4
hidden_layer = 3
output_layer = 3
n_epoch = 300

# inisialisasi network dengan 4 input layer, 3 hidden layer, dan 3 output layer
network = initialize(input_layer, hidden_layer, output_layer)
# learning rate = 0.1
train_data(network, train, val, 0.1, n_epoch, output_layer)

# grafik akurasi
x = plt.figure()
plt.suptitle('Grafik Akurasi Learning Rate 0.1')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(acc_train, '-y')
plt.plot(acc_val, '-b')

plt.gca().legend(('Akurasi data latih', 'Akurasi data validasi'))

# grafik error
y = plt.figure()
y.suptitle('Grafik Error Learning Rate 0.1')
plt.xlabel('epoch')
plt.ylabel('error')
plt.plot(error_train, '-y')
plt.plot(error_val, '-b')

plt.gca().legend(('Error data latih', 'Error data validasi'))

# untuk learning rate 0.8
train_data(network, train, val, 0.8, n_epoch, output_layer)

# grafik akurasi
x = plt.figure()
plt.suptitle('Grafik Akurasi Learning Rate 0.8')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(acc_train, '-y')
plt.plot(acc_val, '-b')

plt.gca().legend(('Akurasi data latih', 'Akurasi data validasi'))

# grafik error
y = plt.figure()
y.suptitle('Grafik Error Learning Rate 0.8')
plt.xlabel('epoch')
plt.ylabel('error')
plt.plot(error_train, '-y')
plt.plot(error_val, '-b')

plt.gca().legend(('Error data latih', 'Error data validasi'))
