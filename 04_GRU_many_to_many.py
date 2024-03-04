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

# Hyper parameters
N, hidden_size, o_size = vocab_size, 100, vocab_size # Hidden size is set to vocab_size, assuming that level of abstractness is approximately proportional to vocab_size (but can be set to any other value).
seq_length = 25 # Longer sequence lengths allow for lengthier latent dependencies to be trained.
learning_rate = 1e-1


##### Recurrent Neural Network Class #####
class GRU:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, num_epochs = 1000):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.hidden_size   = hidden_size
        self.num_epochs    = num_epochs

        # Model parameter initialization
        # Update gate
        self.Wz = np.random.rand(hidden_size, N) * 0.1 - 0.05
        self.Uz = np.random.rand(hidden_size, hidden_size) * 0.1 - 0.05
        self.bz = np.zeros((hidden_size, 1))

        # Reset gate
        self.Wr = np.random.rand(hidden_size, N) * 0.1 - 0.05
        self.Ur = np.random.rand(hidden_size, hidden_size) * 0.1 - 0.05
        self.br = np.zeros((hidden_size, 1))

        self.Wg = np.random.rand(hidden_size, N) * 0.1 - 0.05
        self.Ug = np.random.rand(hidden_size, hidden_size) * 0.1 - 0.05
        self.bg = np.zeros((hidden_size, 1))

        self.Wy = np.random.rand(o_size, hidden_size) * 0.1 - 0.05
        self.by  = np.zeros((vocab_size, 1))   # output bias

    # Forward Propogation
    def forward(self, inputs, targets, h_prev):
        # Initialize variables

        xs, z, r, g, hs, ys, ps = {}, {}, {}, {}, {-1: h_prev}, {}, {} # Dictionaries contain variables for each timestep.
        sequence_loss = 0

        # Forward prop
        for q in range(len(inputs)):
            # Set up one-hot encoded input
            xs[q] = np.zeros((vocab_size, 1))
            xs[q][inputs[q]] = 1

            # Calculate update and reset gates
            z[q] = sigmoid(np.dot(self.Wz, xs[q]) + np.dot(self.Uz, hs[q-1]) + self.bz)
            r[q] = sigmoid(np.dot(self.Wr, xs[q]) + np.dot(self.Ur, hs[q-1]) + self.br)

            # Calculate hidden units
            g[q] = tanh(np.dot(self.Wg, xs[q]) + np.dot(self.Ug, np.multiply(r[q], hs[q-1])) + self.bg)
            hs[q] = np.multiply(z[q], hs[q-1]) + np.multiply((1 - z[q]), g[q])

            # Regular output unit
            ys[q] = np.dot(self.Wy, hs[q]) + self.by

            # Probability distribution
            ps[q] = softmax(ys[q])

            # Cross-entropy loss
            loss = -np.sum(np.log(ps[q][targets[q]]))
            sequence_loss += loss
        return sequence_loss, hs, ps, xs, z, r, g

    def backward(self, inputs, targets, hs, ps, xs, z, r, g):
        # Parameter gradient initialization
        dWy, dWg, dWr, dWz = np.zeros_like(self.Wy), np.zeros_like(self.Wg), np.zeros_like(self.Wr), np.zeros_like(self.Wz)
        dUg, dUr, dUz = np.zeros_like(self.Ug), np.zeros_like(self.Ur), np.zeros_like(self.Uz)
        dby, dbg, dbr, dbz = np.zeros_like(self.by), np.zeros_like(self.bg), np.zeros_like(self.br), np.zeros_like(self.bz)
        dh_next = np.zeros_like(hs[0])

        # Backward Propogation
        for q in reversed(range(len(inputs))):
            # ∂loss/∂y
            dy = np.copy(ps[q])
            dy[targets[q]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here

            # ∂loss/∂Wy and ∂loss/∂by
            dWy += np.dot(dy, hs[q].T)
            dby += dy

            # Intermediary derivatives
            dh   = np.dot(self.Wy.T, dy) + dh_next

            dg   = np.multiply(dh, (1 - z[q]))
            dg_l = dg * tanh(g[q], derivative=True)

            # ∂loss/∂Wg, ∂loss/∂Ug and ∂loss/∂bg
            dWg += np.dot(dg_l, xs[q].T)
            dUg += np.dot(dg_l, np.multiply(r[q], hs[q-1]).T)
            dbg += dg_l

            # Intermediary derivatives
            drhp = np.dot(self.Ug.T, dg_l)
            dr   = np.multiply(drhp, hs[q-1])
            dr_l = dr * sigmoid(r[q], derivative=True)

            # ∂loss/∂Wr, ∂loss/∂Ur and ∂loss/∂br
            dWr += np.dot(dr_l, xs[q].T)
            dUr += np.dot(dr_l, hs[q-1].T)
            dbr += dr_l

            # Intermediary derivatives
            dz   = np.multiply(dh, hs[q-1] - g[q])
            dz_l = dz * sigmoid(z[q], derivative=True)

            # ∂loss/∂Wz, ∂loss/∂Uz and ∂loss/∂bz
            dWz += np.dot(dz_l, xs[q].T)
            dUz += np.dot(dz_l, hs[q-1].T)
            dbz += dz_l

            # All influences of previous layer to loss
            dh_fz_inner = np.dot(self.Uz.T, dz_l)
            dh_fz  = np.multiply(dh, z[q])
            dh_fhh = np.multiply(drhp, r[q])
            dh_fr  = np.dot(self.Ur.T, dr_l)

            # ∂loss/∂h𝑡₋₁
            dh_next = dh_fz_inner + dh_fz + dh_fhh + dh_fr

        return dWy, dWg, dWr, dWz, dUg, dUr, dUz, dby, dbg, dbr, dbz, hs[len(inputs) - 1]

    # Train
    def train(self, data):
        # Initialize sampling parameters and memory gradients (for adagrad)
        n, p = 0, 0
        mdWy, mdWg, mdWr, mdWz = np.zeros_like(self.Wy), np.zeros_like(self.Wg), np.zeros_like(self.Wr), np.zeros_like(self.Wz)
        mdUg, mdUr, mdUz       = np.zeros_like(self.Ug), np.zeros_like(self.Ur), np.zeros_like(self.Uz)
        mdby, mdbg, mdbr, mdbz = np.zeros_like(self.by), np.zeros_like(self.bg), np.zeros_like(self.br), np.zeros_like(self.bz)

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

            # Get gradients for current model based on input and target sequences
            loss, hs, ps, xs, z, r, g = self.forward(inputs, targets, h_prev)
            dWy, dWg, dWr, dWz, dUg, dUr, dUz, dby, dbg, dbr, dbz, h_prev = self.backward(inputs, targets, hs, ps, xs, z, r, g)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # Occasionally print loss information
            if (n+1) % print_interval == 0:
                print('iter %d, loss: %f, smooth loss: %f' % (n+1, loss, smooth_loss))

            # Update model with adagrad (stochastic) gradient descent
            for param, dparam, mem in zip([self.Wy,  self.Wg,  self.Wr,  self.Wz,  self.Ug,  self.Ur,  self.Uz,  self.by,  self.bg,  self.br,  self.bz],
                                          [dWy, dWg, dWr, dWz, dUg, dUr, dUz, dby, dbg, dbr, dbz],
                                          [mdWy,mdWg,mdWr,mdWz,mdUg,mdUr,mdUz,mdby,mdbg,mdbr,mdbz]):
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
        ixes = [seed_ix]

        for q in range(n):
            # Calculate update and reset gates
            z = sigmoid(np.dot(self.Wz, x) + np.dot(self.Uz, h) + self.bz)
            r = sigmoid(np.dot(self.Wr, x) + np.dot(self.Ur, h) + self.br)

            # Calculate hidden units
            g = tanh(np.dot(self.Wg, x) + np.dot(self.Ug, np.multiply(r, h)) + self.bg)
            h = np.multiply(z, h) + np.multiply((1 - z), g)

            # Regular output unit
            y = np.dot(self.Wy, h) + self.by

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

rnn = GRU(input_size = vocab_size, hidden_size = hidden_size, output_size = vocab_size, learning_rate = 0.02, num_epochs = 1000)

##### Training #####
rnn.train(data)