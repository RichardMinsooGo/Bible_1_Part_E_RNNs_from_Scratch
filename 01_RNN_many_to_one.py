##### Imports #####
from tqdm import tqdm
import numpy as np

##### Data #####
"""
train_X = ['good', 'bad', 'happy', 'sad', 'not good', 'not bad', 'not happy', 'not sad', 'very good', 'very bad', 'very happy', 'very sad', 'i am happy', 'this is good', 'i am bad', 'this is bad', 'i am sad', 'this is sad', 'i am not happy', 'this is not good', 'i am not bad', 'this is not sad', 'i am very happy', 'this is very good', 'i am very bad', 'this is very sad', 'this is very happy', 'i am good not bad', 'this is good not bad', 'i am bad not good', 'i am good and happy', 'this is not good and not happy', 'i am not at all good', 'i am not at all bad', 'i am not at all happy', 'this is not at all sad', 'this is not at all happy', 'i am good right now', 'i am bad right now', 'this is bad right now', 'i am sad right now', 'i was good earlier', 'i was happy earlier', 'i was bad earlier', 'i was sad earlier', 'i am very bad right now', 'this is very good right now', 'this is very sad right now', 'this was bad earlier', 'this was very good earlier', 'this was very bad earlier', 'this was very happy earlier', 'this was very sad earlier', 'i was good and not bad earlier', 'i was not good and not happy earlier', 'i am not at all bad or sad right now', 'i am not at all good or happy right now', 'this was not happy and not good earlier']
train_y = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]

test_X = ['this is happy', 'i am good', 'this is not happy', 'i am not good', 'this is not bad', 'i am not sad', 'i am very good', 'this is very bad', 'i am very sad', 'this is bad not good', 'this is good and happy', 'i am not good and not happy', 'i am not at all sad', 'this is not at all good', 'this is not at all bad', 'this is good right now', 'this is sad right now', 'this is very bad right now', 'this was good earlier', 'i was not happy and not good earlier']
test_y = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
"""
train_X = ['this is happy', 'i am good', 'this is not happy', 'not good']
train_y = [1, 1, 0, 0]

chars = set([q for text in train_X for q in text.split()])
vocab_size = len(chars)

char_to_idx = {ch:i for i, ch in enumerate(chars)}
idx_to_char = {i:ch for i, ch in enumerate(chars)}

##### Helper Functions #####
def oneHotEncode(text):
    inputs = []
    for q in text.split():
        vector = np.zeros((1, vocab_size))
        vector[0][char_to_idx[q]] = 1
        inputs += [vector]

    return inputs

# Xavier Normalized Initialization
def initWeights(input_size, output_size):
    return np.random.uniform(-1, 1, (input_size, output_size)) * np.sqrt(6 / (input_size + output_size))

##### Activation Functions #####
def tanh(input, derivative = False):
    if derivative:
        return 1 - input ** 2
    else:
        return np.tanh(input)

# Derivative is directly calculated in backprop (in combination with cross-entropy loss function).
def softmax(input):
    # Subtraction of max value improves numerical stability.
    e_input = np.exp(input - np.max(input))
    return e_input / e_input.sum()

##### Recurrent Neural Network Class #####
class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, num_epochs = 1000):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.hidden_size   = hidden_size
        self.num_epochs    = num_epochs

        # Network parameter initialization
        self.Whx = initWeights(input_size, hidden_size)
        self.Whh = initWeights(hidden_size, hidden_size)
        self.Wyh = initWeights(hidden_size, output_size)

        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))

    # Forward Propogation
    def forward(self, inputs):
        self.hidden_states = [np.zeros_like(self.bh)]

        # print("inputs :", inputs)
        # print("hidden_states len :", len( self.hidden_states) )

        for input in inputs:

            # print("input :", input)

            layer1_output = np.dot(input, self.Whx)
            # print("layer1_output :", layer1_output)

            layer2_output = np.dot(self.hidden_states[-1], self.Whh) + self.bh
            # print("layer2_output :", layer2_output)

            self.hidden_states += [tanh(layer1_output + layer2_output)]
            # print("hidden_states :", self.hidden_states)


        # print("hidden_states len :", len( self.hidden_states) )
        # print("hidden_states :", self.hidden_states )
        forward_out = np.dot(self.hidden_states[-1], self.Wyh) + self.by

        # print("hidden_states len :", len( self.hidden_states) )

        return np.dot(self.hidden_states[-1], self.Wyh) + self.by

    # Backward Propogation
    def backward(self, errors, inputs):
        d_by  = errors
        d_Wyh = np.dot(self.hidden_states[-1].T, errors)

        d_bh  = np.zeros_like(self.bh)
        d_Whh = np.zeros_like(self.Whh)
        d_Whx = np.zeros_like(self.Whx)

        d_hidden_state = np.dot(errors, self.Wyh.T)
        
        for q in reversed(range(len(inputs))):
            d_hidden_state *= tanh(self.hidden_states[q + 1], derivative = True)

            d_bh  += d_hidden_state

            d_Whh += np.dot(self.hidden_states[q].T, d_hidden_state)

            d_Whx += np.dot(inputs[q].T, d_hidden_state)

            d_hidden_state = np.dot(d_hidden_state, self.Whh)

        for d_ in (d_by, d_Wyh, d_bh, d_Whh, d_Whx):
            np.clip(d_, -1, 1, out = d_)

        return d_by, d_Wyh, d_bh, d_Whh, d_Whx


    # Train
    def train(self, inputs, labels):
        for _ in tqdm(range(self.num_epochs)):
            for input, label in zip(inputs, labels):
                input = oneHotEncode(input)

                prediction = self.forward(input)
                # print("prediction :", prediction)

                error = -softmax(prediction)
                error[0][label] += 1

                # print("error :", error)
                d_by, d_Wyh, d_bh, d_Whh, d_Whx = self.backward(error, input)

                self.by  += self.learning_rate * d_by
                self.Wyh += self.learning_rate * d_Wyh
                self.bh  += self.learning_rate * d_bh
                self.Whh += self.learning_rate * d_Whh
                self.Whx += self.learning_rate * d_Whx

    # Test
    def test(self, inputs, labels):
        accuracy = 0
        for input, label in zip(inputs, labels):
            print(input)

            input = oneHotEncode(input)
            prediction = self.forward(input)

            print(['Negative', 'Positive'][np.argmax(prediction)], end = '\n\n')
            if np.argmax(prediction) == label:
                accuracy += 1

        
        print(f'Accuracy: {round(accuracy * 100 / len(inputs), 2)}%')
        
# Initialize Network
hidden_size = 9

rnn = RNN(input_size = vocab_size, hidden_size = hidden_size, output_size = 2, learning_rate = 0.02, num_epochs = 1000)

##### Training #####
rnn.train(train_X, train_y)

##### Testing #####
# rnn.test(test_X, test_y)
rnn.test(train_X, train_y)
