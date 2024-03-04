import numpy as np

# Seed random
np.random.seed(0)

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

# Hyper parameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1


##### Recurrent Neural Network Class #####
class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, num_epochs = 1000):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.hidden_size   = hidden_size
        self.num_epochs    = num_epochs
        
        # Network parameter initialization
        self.Wxh = initWeights(hidden_size, vocab_size)   # input to hidden
        self.Whh = initWeights(hidden_size, hidden_size)  # hidden to hidden
        self.Why = initWeights(vocab_size, hidden_size)   # hidden to output
        
        self.bh  = np.zeros((hidden_size, 1))  # hidden bias
        self.by  = np.zeros((vocab_size, 1))   # output bias

    # Forward Propogation
    def forward(self, inputs, targets, h_prev):
        # Initialize variables

        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        sequence_loss = 0

        # Forward prop
        for q in range(len(inputs)):
            # Set up one-hot encoded input
            xs[q] = np.zeros((vocab_size, 1))
            xs[q][inputs[q]] = 1
            hs[q] = np.tanh(np.dot(self.Wxh, xs[q]) + np.dot(self.Whh, hs[q-1]) + self.bh)   # hidden state
            ys[q] = np.dot(self.Why, hs[q]) + self.by                 # unnormalized log probabilities for next chars

            # Probability distribution
            ps[q] = np.exp(ys[q]) / np.sum(np.exp(ys[q]))   # probabilities for next chars

            # Cross-entropy loss
            loss = -np.log(ps[q][targets[q],0])            # softmax (cross-entropy loss)
            sequence_loss += loss
        return sequence_loss, hs, ps, xs

    # Backward Propogation
    def backward(self, inputs, targets, hs, ps, xs):
        # Parameter gradient initialization
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])

        # Backward Propogation
        for q in reversed(range(len(inputs))):
            # ∂loss/∂y
            dy = np.copy(ps[q])
            dy[targets[q]] -= 1               # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here

            dWhy += np.dot(dy, hs[q].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dh_next   # backprop into h
            dhraw = (1 - hs[q] * hs[q]) * dh  # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[q].T)
            dWhh += np.dot(dhraw, hs[q-1].T)
            dh_next = np.dot(self.Whh.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
        return dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

    # Train
    def train(self, data):
        # Initialize sampling parameters and memory gradients (for adagrad)
        n, p = 0, 0
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        mbh, mby         = np.zeros_like(self.bh), np.zeros_like(self.by)    # memory variables for Adagrad
        smooth_loss      = -np.log(1.0/vocab_size)*seq_length      # loss at iteration 0

        print_interval = 100

        for epoch in range(10000):
            # Reset memory if appropriate
            if p+seq_length+1 >= len(data) or n == 0: 
                h_prev = np.zeros((hidden_size,1)) # reset RNN memory
                p = 0 # go from start of data

            # Get input and target sequence
            inputs  = [char_to_idx[ch] for ch in data[p:p+seq_length]]
            targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]

            # Occasionally sample from model and print result
            if n % print_interval == 0:
                sample_ix = self.test(h_prev, inputs[0], data_size)
                txt = ''.join(idx_to_char[ix] for ix in sample_ix)
                print('----\n%s\n----' % (txt, ))

            # forward seq_length characters through the net and fetch gradient
            loss, hs, ps, xs = self.forward(inputs, targets, h_prev)
            dWxh, dWhh, dWhy, dbh, dby, h_prev = self.backward(inputs, targets, hs, ps, xs)

            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # Occasionally print loss information
            if (n+1) % print_interval == 0:
                print('iter %d, loss: %f, smooth loss: %f' % (n+1, loss, smooth_loss))

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                          [dWxh, dWhh, dWhy, dbh, dby], 
                                          [mWxh, mWhh, mWhy, mbh, mby]):
                np.clip(dparam, -5, 5, out=dparam)
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

            # Prepare for next iteration
            p += seq_length # move data pointer
            n += 1 # iteration counter
    
    def test(self, h, seed_ix, n):
        # Initialize first word of sample ('seed') as one-hot encoded vector.
        x = np.zeros((vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for q in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by

            # Probability distribution
            p = softmax(y)

            # Choose next char according to the distribution
            ix = np.random.choice(range(vocab_size), p=p.ravel())
            x = np.zeros((vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)

        return ixes
        
# Initialize Network
hidden_size = vocab_size

rnn = RNN(input_size = vocab_size, hidden_size = hidden_size, output_size = vocab_size, learning_rate = 0.02, num_epochs = 1000)

##### Training #####
rnn.train(data)