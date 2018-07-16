import torch

class LSTM(torch.nn.Module):
    def __init__(self, parameters):
        """
        One layer LSTM network.
        Arguments:
            parameters -- python dictionary containing:
                Wf -- Weight matrix of the forget gate, of shape (n_a, n_a + n_x)
                bf -- Bias of the forget gate, of shape (n_a, 1)
                Wi -- Weight matrix of the update gate, of shape (n_a, n_a + n_x)
                bi -- Bias of the update gate, of shape (n_a, 1)
                Wc -- Weight matrix of the first "tanh", of shape (n_a, n_a + n_x)
                bc --  Bias of the first "tanh", of shape (n_a, 1)
                Wo -- Weight matrix of the output gate, of shape (n_a, n_a + n_x)
                bo --  Bias of the output gate, of shape (n_a, 1)
                Wy -- Weight matrix relating the hidden-state to the output, of shape (n_y, n_a)
                by -- Bias relating the hidden-state to the output, of shape (n_y, 1)
        """
        super(LSTM, self).__init__()

        # Retrieve parameters from "parameters"
        Wf = parameters["Wf"]
        bf = parameters["bf"]
        Wi = parameters["Wi"]
        bi = parameters["bi"]
        Wc = parameters["Wc"]
        bc = parameters["bc"]
        Wo = parameters["Wo"]
        bo = parameters["bo"]
        Wy = parameters["Wy"]
        by = parameters["by"]

    def lstm_cell_forward(self, xt, a_prev, c_prev):
        """
        Implement a single forward step of the LSTM-cell

        Arguments:
        xt -- your input data at timestep "t", of shape (n_x, m).
        a_prev -- Hidden state at timestep "t-1", of shape (n_a, m)
        c_prev -- Memory state at timestep "t-1", of shape (n_a, m)

        Returns:
        a_next -- next hidden state, of shape (n_a, m)
        c_next -- next memory state, of shape (n_a, m)
        yt_pred -- prediction at timestep "t", of shape (n_y, m)

        Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
              c stands for the memory value
        """

        # Retrieve dimensions from shapes of xt and Wy
        n_x, m = xt.size()
        n_y, n_a = Wy.size()

        # Concatenate a_prev and xt
        concat = torch.zeros(n_a + n_x, m, device=device, dtype=dtype)
        concat[:n_a, :] = a_prev
        concat[n_a:, :] = xt

        # Compute values for ft, it, cct, c_next, ot, a_next
        tanh = torch.nn.Tanh()
        ft = torch.sigmoid(Wf.mm(concat) + bf.expand(n_a, m))
        it = torch.sigmoid(Wi.mm(concat) + bi.expand(n_a, m))
        cct = tanh(Wc.mm(concat) + bc.expand(n_a, m))
        c_next = it * cct + c_prev.clone() * ft
        ot = torch.sigmoid(Wo.mm(concat) + bo.expand(n_a, m))
        a_next = ot * tanh(c_next)

        # Compute prediction of the LSTM cell
        softmax = torch.nn.Softmax(dim=0)
        yt_pred = softmax(Wy.mm(a_next) + by.expand(n_y, m))

        return a_next, c_next, yt_pred

    def forward(self, x, a0):
        """
        Implement the forward propagation of the recurrent neural network using an LSTM-cell.

        Arguments:
        x -- Input data for every time-step, of shape (n_x, m, T_x).
        a0 -- Initial hidden state, of shape (n_a, m)

        Returns:
        y -- Predictions for every time-step, of shape (n_y, m, T_x)
        """

        # Retrieve dimensions from shapes of x and Wy
        n_x, m, T_x = x.size()
        n_y, n_a = Wy.size()

        # initialize "a", "c" and "y" with zeros
        a = torch.zeros(n_a, m, T_x + 1, device=device, dtype=dtype)
        a[:, :, 0] = a0
        c = torch.zeros(n_a, m, T_x + 1, device=device, dtype=dtype)
        y = torch.zeros(n_y, m, T_x, device=device, dtype=dtype)

        # loop over all time-steps
        for t in range(1, T_x + 1):
            # Update next hidden state, next memory state, compute the prediction
            a[:, :, t], c[:, :, t], y[:, :, t-1] = self.lstm_cell_forward(x[:, :, t-1], a[:, :, t-1], c[:, :, t-1])

        return y

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
dtype = torch.float

(n_x, n_a, n_y, T_x, m) = (3, 5, 2, 7, 10)

x = torch.randn(n_x, m, T_x, device=device, dtype=dtype)
a0 = torch.randn(n_a, m, device=device, dtype=dtype)

y = torch.randint(0, n_y, (m, T_x), device=device, dtype=torch.long).unsqueeze(0)
y_onehot = torch.zeros(n_y, m, T_x, device=device, dtype=dtype)
y_onehot.scatter_(0, y, 1)

# parameters
Wf = torch.randn(n_a, n_a+n_x, device=device, dtype=dtype, requires_grad=True)
bf = torch.randn(n_a, 1, device=device, dtype=dtype, requires_grad=True)
Wi = torch.randn(n_a, n_a+n_x, device=device, dtype=dtype, requires_grad=True)
bi = torch.randn(n_a, 1, device=device, dtype=dtype, requires_grad=True)
Wo = torch.randn(n_a, n_a+n_x, device=device, dtype=dtype, requires_grad=True)
bo = torch.randn(n_a, 1, device=device, dtype=dtype, requires_grad=True)
Wc = torch.randn(n_a, n_a+n_x, device=device, dtype=dtype, requires_grad=True)
bc = torch.randn(n_a, 1, device=device, dtype=dtype, requires_grad=True)
Wy = torch.randn(n_y, n_a, device=device, dtype=dtype, requires_grad=True)
by = torch.randn(n_y, 1, device=device, dtype=dtype, requires_grad=True)
parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

learning_rate = 1e-4
num_iters = 500

model = LSTM(parameters)
loss_fn = torch.nn.BCELoss(size_average=False)

for t in range(num_iters):
    y_pred = model(x, a0) # forward pass
    loss = loss_fn(y_pred, y_onehot)

    # backward pass
    loss.backward()

    with torch.no_grad():
        for paramNames, params in parameters.items():
            params -= learning_rate * params.grad
            params.grad.zero_()

print("Final Loss: ", loss.item())
