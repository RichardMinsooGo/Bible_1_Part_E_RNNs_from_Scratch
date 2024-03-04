##### Imports #####
from tqdm import tqdm
import numpy as np

##### Data #####

data_old = """The Dursleys had everything they wanted, but they also had a secret, and \
their greatest fear was that somebody would discover it. They didn't \
think they could bear it if anyone found out about the Potters. Mrs. \
Potter was Mrs. Dursley's sister, but they hadn't met for several years; \
in fact, Mrs. Dursley pretended she didn't have a sister, because her \
sister and her good-for-nothing husband were as unDursleyish as it was \
possible to be. The Dursleys shuddered to think what the neighbors would \
say if the Potters arrived in the street. The Dursleys knew that the \
Potters had a small son, too, but they had never even seen him. This boy \
was another good reason for keeping the Potters away; they didn't want \
Dudley mixing with a child like that.""".lower()

data = """The Dursleys had everything they wanted, but they also had a secret, and \
their greatest fear was that somebody would discover it.""".lower()

chars = set(data)

data_size, vocab_size = len(data), len(chars)

print('data has %d characters, %d unique.' % (data_size, vocab_size))

char_to_idx = {ch:i for i, ch in enumerate(chars)}
idx_to_char = {i:ch for i, ch in enumerate(chars)}

train_X, train_y = data[:-1], data[1:]

##### Helper Functions #####
def oneHotEncode(text):
    output = np.zeros((vocab_size, 1))
    output[char_to_idx[text]] = 1

    return output

# Xavier Normalized Initialization
def initWeights(input_size, output_size):
    return np.random.uniform(-1, 1, (output_size, input_size)) * np.sqrt(6 / (input_size + output_size))

##### Activation Functions #####
def sigmoid(input, derivative = False):
    if derivative:
        return input * (1 - input)
    else:
        return 1 / (1 + np.exp(-input))

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

##### Long Short-Term Memory Network Class #####
class LSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, num_epochs = 1000):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.hidden_size   = hidden_size
        self.num_epochs    = num_epochs
        
        # Forget Gate
        self.Wf = initWeights(input_size, hidden_size)
        self.bf = np.zeros((hidden_size, 1))

        # Input Gate
        self.Wi = initWeights(input_size, hidden_size)
        self.bi = np.zeros((hidden_size, 1))

        # Candidate Gate
        self.Wc = initWeights(input_size, hidden_size)
        self.bc = np.zeros((hidden_size, 1))

        # Output Gate
        self.Wo = initWeights(input_size, hidden_size)
        self.bo = np.zeros((hidden_size, 1))

        # Final Gate
        self.Wy = initWeights(hidden_size, output_size)
        self.by = np.zeros((output_size, 1))

    # Reset Network Memory
    def reset(self):
        self.concat_inputs = {}

        self.hidden_states = {-1:np.zeros((self.hidden_size, 1))}
        self.cell_states   = {-1:np.zeros((self.hidden_size, 1))}

        self.activation_outputs = {}
        self.candidate_memos    = {}
        self.output_gates       = {}
        self.forget_gates       = {}
        self.input_gates        = {}
        self.outputs            = {}

    # Forward Propogation
    def forward(self, inputs):
        self.reset()

        outputs = []
        for q in range(len(inputs)):
            self.concat_inputs[q]   = np.concatenate((self.hidden_states[q - 1], inputs[q]))

            self.forget_gates[q]    = sigmoid(np.dot(self.Wf, self.concat_inputs[q]) + self.bf)
            self.input_gates[q]     = sigmoid(np.dot(self.Wi, self.concat_inputs[q]) + self.bi)
            self.candidate_memos[q] = tanh(np.dot(self.Wc, self.concat_inputs[q])    + self.bc)
            self.output_gates[q]    = sigmoid(np.dot(self.Wo, self.concat_inputs[q]) + self.bo)

            self.cell_states[q]     = self.forget_gates[q] * self.cell_states[q - 1] + self.input_gates[q] * self.candidate_memos[q]
            self.hidden_states[q]   = self.output_gates[q] * tanh(self.cell_states[q])

            outputs += [np.dot(self.Wy, self.hidden_states[q]) + self.by]

        return outputs

    # Backward Propogation
    def backward(self, errors, inputs):
        d_wf, d_bf = 0, 0
        d_wi, d_bi = 0, 0
        d_wc, d_bc = 0, 0
        d_wo, d_bo = 0, 0
        d_wy, d_by = 0, 0

        dh_next, dc_next = np.zeros_like(self.hidden_states[0]), np.zeros_like(self.cell_states[0])
        for q in reversed(range(len(inputs))):
            error = errors[q]

            # Final Gate Weights and Biases Errors
            d_wy += np.dot(error, self.hidden_states[q].T)
            d_by += error

            # Hidden State Error
            d_hs = np.dot(self.Wy.T, error) + dh_next

            # Output Gate Weights and Biases Errors
            d_o = tanh(self.cell_states[q]) * d_hs * sigmoid(self.output_gates[q], derivative = True)
            d_wo += np.dot(d_o, inputs[q].T)
            d_bo += d_o

            # Cell State Error
            d_cs = tanh(tanh(self.cell_states[q]), derivative = True) * self.output_gates[q] * d_hs + dc_next

            # Forget Gate Weights and Biases Errors
            d_f = d_cs * self.cell_states[q - 1] * sigmoid(self.forget_gates[q], derivative = True)
            d_wf += np.dot(d_f, inputs[q].T)
            d_bf += d_f

            # Input Gate Weights and Biases Errors
            d_i = d_cs * self.candidate_memos[q] * sigmoid(self.input_gates[q], derivative = True)
            d_wi += np.dot(d_i, inputs[q].T)
            d_bi += d_i
            
            # Candidate Gate Weights and Biases Errors
            d_c = d_cs * self.input_gates[q] * tanh(self.candidate_memos[q], derivative = True)
            d_wc += np.dot(d_c, inputs[q].T)
            d_bc += d_c

            # Concatenated Input Error (Sum of Error at Each Gate!)
            d_z = np.dot(self.Wf.T, d_f) + np.dot(self.Wi.T, d_i) + np.dot(self.Wc.T, d_c) + np.dot(self.Wo.T, d_o)

            # Error of Hidden State and Cell State at Next Time Step
            dh_next = d_z[:self.hidden_size, :]
            dc_next = self.forget_gates[q] * d_cs

        for d_ in (d_wf, d_bf, d_wi, d_bi, d_wc, d_bc, d_wo, d_bo, d_wy, d_by):
            np.clip(d_, -1, 1, out = d_)

        return d_wf, d_bf, d_wi, d_bi, d_wc, d_bc, d_wo, d_bo, d_wy, d_by

    # Train
    def train(self, inputs, labels):
        inputs = [oneHotEncode(input) for input in inputs]

        for _ in tqdm(range(self.num_epochs)):
            predictions = self.forward(inputs)
            
            errors = []
            for q in range(len(predictions)):
                errors += [-softmax(predictions[q])]
                errors[-1][char_to_idx[labels[q]]] += 1

            d_wf, d_bf, d_wi, d_bi, d_wc, d_bc, d_wo, d_bo, d_wy, d_by = self.backward(errors, self.concat_inputs)
            
            self.Wf += d_wf * self.learning_rate
            self.bf += d_bf * self.learning_rate

            self.Wi += d_wi * self.learning_rate
            self.bi += d_bi * self.learning_rate

            self.Wc += d_wc * self.learning_rate
            self.bc += d_bc * self.learning_rate

            self.Wo += d_wo * self.learning_rate
            self.bo += d_bo * self.learning_rate

            self.Wy += d_wy * self.learning_rate
            self.by += d_by * self.learning_rate
            
    # Test
    def test(self, inputs, labels):
        accuracy = 0
        probabilities = self.forward([oneHotEncode(input) for input in inputs])

        output = ''
        for q in range(len(labels)):
            prediction = idx_to_char[np.random.choice([*range(vocab_size)], p = softmax(probabilities[q].reshape(-1)))]

            output += prediction

            if prediction == labels[q]:
                accuracy += 1

        print(f'Ground Truth:\nt{labels}\n')
        print(f'Predictions:\nt{"".join(output)}\n')
        
        print(f'Accuracy: {round(accuracy * 100 / len(inputs), 2)}%')
        
# Initialize Network
hidden_size = vocab_size

lstm = LSTM(input_size = vocab_size + hidden_size, hidden_size = hidden_size, output_size = vocab_size, learning_rate = 0.05, num_epochs = 1000)

##### Training #####
lstm.train(train_X, train_y)

##### Testing #####
lstm.test(train_X, train_y)
