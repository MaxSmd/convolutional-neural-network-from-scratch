import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import numba

class Layer:
    def __init__(self):
        self.last_input = None

    def forward(self, input_):
        pass

    def backward(self, grad_out, learning_rate):
        pass


class Conv2D(Layer):
    def __init__(self, num_filters, kernel_size, stride, padding):
        """
        Args:
            num_filters (int): the number of filters in the convolution layer
            kernel_size (int): integer specifying the size of the square convolution kernel
            stride (int): integer specifying the stride length of the convolution
            padding (string): can either be "valid"
        """
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.filters = None
        self.stride = stride
        self.padding = padding

    def forward(self, input_):
        """
        Args:
            input_ (np.array): the input tensor with shape (batch_size, height, width, channels)
        """
        self.last_input = input_
        if self.filters is None:
            self.filters = np.random.randn(self.num_filters, self.kernel_size, self.kernel_size, input_.shape[-1]) * 0.1

        outputs = []
        for picture in range(input_.shape[0]):
            if self.padding == 'valid':
                outputs.append(self._forward_valid(input_[picture]))
            else:
                raise ValueError(f'Padding is {self.padding} but must be "valid"')

        return np.array(outputs)

    def _forward_valid(self, picture) -> np.array:
        height_input, width_input, channels_input = picture.shape
        height_output = (height_input - self.kernel_size) // self.stride + 1
        width_output = (width_input - self.kernel_size) // self.stride + 1

        picture_output = np.zeros((height_output, width_output, self.num_filters))
        for h in range(height_output):
            for w in range(width_output):
                window = picture[
                    h * self.stride: h * self.stride + self.kernel_size,
                    w * self.stride: w * self.stride + self.kernel_size,
                    :
                ]

                for f in range(self.num_filters):
                    picture_output[h, w, f] = np.sum(window * self.filters[f])

        return picture_output


    def backward(self, grad_out, learning_rate):
        """
        Args:
            grad_out (np.array): gradient from next layer of shape
                                 (batch_size, out_height, out_width, num_filters)
            learning_rate (float)
        Returns:
            grad_input (np.array): gradient w.r.t the input of shape same as last_input
        """
        batch_size, height_input, width_input, channels = self.last_input.shape

        grad_input = np.zeros_like(self.last_input)
        grad_filters = np.zeros_like(self.filters)

        for b in range(batch_size):
            for h in range(grad_out.shape[1]):
                for w in range(grad_out.shape[2]):
                    for f in range(self.num_filters):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        window = self.last_input[b, h_start: h_end, w_start: w_end, :]

                        # Accumulate gradient w.r.t filter
                        grad_filters[f] += window * grad_out[b, h, w, f]

                        # Accumulate gradient w.r.t input
                        grad_input[b, h_start: h_end, w_start: w_end, :] += self.filters[f] * grad_out[b, h, w, f]

        self.filters -= learning_rate * grad_filters

        return grad_input


class MaxPool2D(Layer):
    def __init__(self, pool_size: int, stride: int):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.max_indices = None

    def forward(self, input_: np.array) -> np.array:
        self.last_input = input_
        batch_size, input_height, input_width, channels = input_.shape
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1

        # Create a view of all possible windows of size pool_size * poolsize on the input_
        # The shape will be (batch_size, input_height, input_width, channels, pool_size, pool_size)
        windows = sliding_window_view(input_, (self.pool_size, self.pool_size), axis=(1, 2))

        # Select only the windows at the specified strides
        windows = windows[:, ::self.stride, ::self.stride, :, :, :]

        # Flatten the 2 dimensional windows to find maximum value in them
        # The shape becomes (batch_size, output_height, output_width, channels, pool_size * pool_size)
        flattened_windows = windows.reshape(batch_size, output_height, output_width, channels, -1)

        # Both max_indices and output have shape (batch_size, output_height, output_width, channels)
        self.max_indices = np.argmax(flattened_windows, axis=-1)
        output = np.max(flattened_windows, axis=-1)

        return output


    def backward(self, grad_out, learning_rate):
        grad_in = np.zeros_like(self.last_input)

        batch_size, input_height, input_width, channels = self.last_input.shape
        _, output_height, output_width, _ = grad_out.shape

        # Convert the flat indices from `argmax` back into 2D (row, col) coordinates within each pooling window
        # Both i and j have shape (batch_size, output_height, output_width, channels)
        i, j = np.divmod(self.max_indices, self.pool_size)

        # Calculate the absolute coordinates in the input feature map by
        # adding the stride offset to the window coordinates (i, j)
        h_idx = (np.arange(output_height)[None, :, None, None] * self.stride + i)
        w_idx = (np.arange(output_width)[None, None, :, None] * self.stride + j)

        # Create broadcast-compatible indices for batch and channel dimensions.
        b_idx = np.arange(batch_size)[:, None, None, None]
        c_idx = np.arange(channels)[None, None, None, :]

        # "Scatter" the incoming gradient to the correct locations. Add the gradient from grad_out to the correct
        # coordinates of grad_in specified by (b_idx, h_idx, w_idx, c_idx)
        np.add.at(grad_in, (b_idx, h_idx, w_idx, c_idx), grad_out)

        return grad_in


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        self.last_input = input_
        return input_.reshape(input_.shape[0], -1)

    def backward(self, grad_out, learning_rate):
        return grad_out.reshape(self.last_input.shape)


class Dense(Layer):
    def __init__(self, input_size, units):
        """
        Args:
        units (int): dimensionality of the output space
        """
        super().__init__()
        self.weights = np.random.randn(input_size, units) * 0.1  # (input_size, units)
        self.bias = np.zeros(units)  # (units,)

    def forward(self, input_):
        """
        Args:
        input_ (np.array): the input of shape (batch_size, input_size)
        """
        self.last_input = input_
        return input_ @ self.weights + self.bias  # (batch_size, units)

    def backward(self, grad_out, learning_rate):
        """
        Args:
        grad_out (np.array): the previous gradient of shape (batch_size, units)
        learning_rate (float)
        """
        grad_in = grad_out @ self.weights.T  # (batch_size, input_size)
        self.bias -= learning_rate * grad_out.sum(axis=0)  # (units,)
        self.weights -= learning_rate * self.last_input.T @ grad_out  # (input_size, units)
        return grad_in


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.last_output = None

    def forward(self, input_):
        self.last_output = self._sigmoid(input_)
        return self.last_output

    def backward(self, grad_out, learning_rate):
        return grad_out * self.last_output * (1.0 - self.last_output)

    @staticmethod
    def _sigmoid(x):
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        z = np.empty_like(x)

        z[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
        exp_x = np.exp(x[neg_mask])
        z[neg_mask] = exp_x / (1.0 + exp_x)
        return z


class ReLu(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out, learning_rate):
        return grad_out * self.mask


class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.last_output = None

    def forward(self, input_):
        exps = np.exp(input_ - np.max(input_, axis=-1, keepdims=True))
        self.last_output = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.last_output

    def backward(self, grad_out, learning_rate):
        dot = np.sum(grad_out * self.last_output, axis=1, keepdims=True)  # (batch_size, 1)
        grad_input = self.last_output * (grad_out - dot)  # (batch_size, num_classes)
        return grad_input







