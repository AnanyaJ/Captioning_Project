import torch

class OneLayerCNN(torch.nn.Module):
    def __init__(self, W, b, hparameters):
        """
        One convolutional layer.

        Arguments:
            W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev, n_C)
            b -- Biases, tensor of shape (1, 1, 1, n_C)
            hparameters -- python dictionary containing "stride" and "pad"
        """
        super(OneLayerCNN, self).__init__()

        self.W = torch.nn.Parameter(W)
        self.b = torch.nn.Parameter(b)

    def zero_pad(self, X, pad):
        """
        Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image.

        Argument:
        X -- python tensor of shape (m, n_H, n_W, n_C) representing a batch of m images
        pad -- integer, amount of padding around each image on vertical and horizontal dimensions

        Returns:
        X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
        """

        pad_fn = torch.nn.ConstantPad3d((0,0,pad,pad,pad,pad), 0)
        X_pad = pad_fn(X)
        return X_pad

    def conv_single_step(self, a_slice_prev):
        """
        Apply filters defined by parameters W on a single slice (a_slice_prev) of the output activation
        of the previous layer.

        Arguments:
        a_slice_prev -- slice of input data of shape (f, f, n_C_prev, n_C)

        Returns:
        Z -- vector of size n_C, result of convolving the sliding window (W, b) on a slice x of the input data
        """

        # Element-wise product between a_slice and W.
        s = a_slice_prev * self.W
        # Sum over all entries of the first three dimensions of s and add biases b for each filter
        Z = s.sum(0).sum(0).sum(0) + self.b.squeeze()

        return Z

    def forward(self, A_prev):
        """
        Implements the forward propagation with a convolutional layer with a ReLU activation

        Arguments:
        A_prev -- output activations of the previous layer, tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        A_unrolled -- output, tensor of shape (m, n_H*n_W*n_C)
        """

        # Retrieve dimensions from A_prev's shape and W's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.size()
        (f, f, n_C_prev, n_C) = self.W.size()

        # Retrieve information from "hparameters"
        stride = hparameters["stride"]
        pad = hparameters["pad"]

        # Compute the dimensions of the CONV output volume
        n_H = int((n_H_prev - f + 2*pad)/stride) + 1
        n_W = int((n_W_prev - f + 2*pad)/stride) + 1

        # Initialize the output volume Z with zeros. (≈1 line)
        Z = torch.zeros(m, n_H, n_W, n_C, device=device, dtype=dtype)

        # Create A_prev_pad by padding A_prev
        A_prev_pad = self.zero_pad(A_prev, pad)

        for i in range(m):  # loop over the batch of training examples
            a_prev_pad = A_prev_pad[i]  # Select ith training example's padded activation
            for h in range(n_H):  # loop over vertical axis of the output volume
                for w in range(n_W):  # loop over horizontal axis of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    # Use the corners to define the (3D) slice of a_prev_pad
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    a_slice_prev = a_slice_prev.unsqueeze(3).expand(f, f, n_C_prev, n_C)

                    # Convolve the (3D) slice with all filters W and biases b, to get back n_C output neurons
                    Z[i, h, w] = self.conv_single_step(a_slice_prev).clamp(min=0)

        A = Z.clamp(min=0) # ReLU activation
        A_unrolled = torch.reshape(A, (m, n_H*n_W*n_C))

        return A_unrolled

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
dtype = torch.float

(m, f, n_H_prev, n_W_prev, n_C_prev, n_H, n_W, n_C) = (10, 2, 4, 4, 3, 4, 4, 5)

x = torch.randn(m, n_H_prev, n_W_prev, n_C_prev, device=device, dtype=dtype)
y = torch.randint(0, 2, (m, 1), device=device, dtype=dtype)
W = torch.randn(f, f, n_C_prev, n_C, device=device, dtype=dtype, requires_grad=True)
b = torch.randn(1, 1, 1, n_C, device=device, dtype=dtype, requires_grad=True)

hparameters = {"pad" : 2, "stride" : 2}
learning_rate = 1e-4
num_iters = 100

model = torch.nn.Sequential(
    OneLayerCNN(W, b, hparameters),
    torch.nn.Linear(n_H * n_W * n_C, 1),
    torch.nn.Sigmoid(),
)

loss_fn = torch.nn.BCELoss(size_average=False) # binary cross entropy loss

for t in range(num_iters):
    y_pred = model(x) # forward pass

    loss = loss_fn(y_pred, y)

    # backward pass
    loss.backward()

    with torch.no_grad():
        for params in model.parameters():
            params -= learning_rate * params.grad

print("Final Loss: ", loss.item())