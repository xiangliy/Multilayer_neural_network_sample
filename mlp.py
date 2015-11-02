''' Feel free to use numpy for matrix multiplication and
    other neat features.
    You can write some helper functions to
    place some calculations outside the other functions
    if you like to.
'''
import numpy as np
import random

iteration = 30
learning_rate = 0.001
hidden_node_num = 12
input_layer_num = 40
output_layer_num = 8

class mlp:
    w_hidden = [[1 for i in range(hidden_node_num + 1)] for j in range(input_layer_num + 1)]
    w_output = [[1 for i in range(output_layer_num)] for j in range(hidden_node_num + 1)]

    def __init__(self, inputs, targets):
        self.beta = 1
        self.eta = 0.1
        self.momentum = 0.0

        #init w_hidden
        for i in range(0, input_layer_num + 1):
            for j in range(0, hidden_node_num + 1):
                self.w_hidden[i][j] = random.uniform(0, 1)

        #init w_hidden
        for i in range(0, hidden_node_num + 1):
            for j in range(0, output_layer_num):
                self.w_output[i][j] = random.uniform(0, 1)

    def earlystopping(self, inputs, targets, valid, validtargets):
        old_error_rate = 1
        test_array = []

        for i in range(0, iteration):
            self.train(inputs, targets)
            error_rate = self.test(valid, validtargets)

            if error_rate > old_error_rate and error_rate > 0.7:
                test_array.append(error_rate)
                #print(i)
                #print(error_rate)
                #break
            else:
                test_array.append(error_rate)
                #print(i)
                #print(error_rate)
                old_error_rate = error_rate

        print test_array

    def train(self, inputs, targets, iterations=10):
        for i in range(0, iterations):
            outputs, o_hiddens, o_inputs = self.forward(inputs)
            #print("---------------------------------------------------------")
            print self.test(inputs, targets)

            for j in range(0, len(outputs)):
                output = np.matrix(outputs[j])
                target = np.matrix(targets[j])

                #calculate delta of output layer and hidden layer
                d_output = target - output
                d_hidden = d_output * np.matrix(self.w_output).transpose()

                w_output_temp = np.matrix(self.w_output).transpose() + learning_rate * np.matrix(d_output).transpose() * o_hiddens[j]
                self.w_output = w_output_temp.transpose().A

                w_hidden_temp = np.matrix(self.w_hidden).transpose() + learning_rate * np.matrix(d_hidden).transpose() * o_inputs[j]
                self.w_hidden = w_hidden_temp.transpose().A

                #print output

                #print(self.w_hidden)

    def forward(self, inputs):
        outputs = []
        o_hiddens = []
        o_inputs = []
        validateoutputs = []

        l = len(inputs)

        for i in range(0, len(inputs)):
            input_array = inputs[i].tolist()
            input_array.append(-1)

            o_inputs.append(input_array)

            #calculate the output of hidden layer
            i_hidden = np.matrix(input_array) * np.matrix(self.w_hidden)
            i_hidden = i_hidden.A1
            o_hidden = self.calhidden(i_hidden)
            o_hidden[hidden_node_num] = -1;

            o_hiddens.append(o_hidden)

            i_output = np.matrix(o_hidden) * np.matrix(self.w_output)
            i_output = i_output.A1
            o_output = self.caloutput(i_output)
            o_output = o_output.tolist()

            outputs.append(o_output)

        return outputs, o_hiddens, o_inputs

    def test(self, inputs, targets):
        outputs, o_hiddens, o_inputs = self.forward(inputs)

        nothit = 0.0
        for i in range(0, len(targets)):
            for j in range(0, len(targets[i])):
                if outputs[i][j] != targets[i][j]:
                    nothit += 1
                    break

        error_rate = nothit / len(targets)
        return 1 - error_rate

    def confusion(self, inputs, targets):
        print self.test(inputs, targets)
        confusion_array = [[0 for i in range(8)] for j in range(8)]
        outputs, o_hiddens, o_inputs = self.forward(inputs)
        count = 0
        for i in range(0, len(inputs)):
            corx = -1
            cory = -1
            for j in range(0, len(outputs[i])):
                if outputs[i][j] == 1:
                    corx = j
                    break

            for j in range(0, len(targets[i])):
                if targets[i][j] == 1:
                    cory = j
                    break
            if corx != -1:
                confusion_array[corx][cory] += 1
            else:
                count += 1

        #print count
        #print np.matrix(confusion_array)
        return confusion_array


    def calhidden(self, value_array):
        for i in range(0, len(value_array)):
            value_array[i] = 1 / (1 + np.exp(-value_array[i]))
        #print value_array
        return value_array

    def caloutput(self, value_array):
        for i in range(0, len(value_array)):
            #for i in range(0, len(value_array)):
                #value_array[i] = 1 / (1 + np.exp(-value_array[i]))

            if value_array[i] > 0.5:
                value_array[i] = 1
            else:
                value_array[i] = 0

        return value_array