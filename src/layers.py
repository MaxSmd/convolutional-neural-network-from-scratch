import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numba import jit
from math import ceil

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
            padding (string): can either be "valid" or "same"
        """
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.filters = None
        self.bias = None
        self.stride = stride

        if padding not in ["valid", "same"]:
            raise ValueError(f"Unsupported padding mode: {padding}. Please use 'valid' or 'same'.")
        self.padding = padding

    def forward(self, input_):
        """
        Args:
            input_ (np.array): the input tensor with shape (batch_size, height, width, channels)
        """
        self.last_input = input_
        channels = input_.shape[-1]
        input_padded = self._pad_input(input_)
        self.last_input_padded = input_padded

        if self.filters is None:
            # He-like initialization
            scale = np.sqrt(2.0 / (self.kernel_size * self.kernel_size * channels))
            self.filters = np.random.randn(self.num_filters, self.kernel_size, self.kernel_size, channels) * scale
            self.bias = np.zeros(self.num_filters)

        return self._forward(input_padded, self.filters, self.bias, self.stride)

    def _pad_input(self, input_: np.array) -> np.array:
        if self.padding == 'same':
            _, input_height, input_width, _ = input_.shape

            output_height = ceil(input_height / self.stride)
            output_width  = ceil(input_width / self.stride)

            height_padded = max((output_height - 1) * self.stride + self.kernel_size - input_height, 0)
            width_padded = max((output_width - 1) * self.stride + self.kernel_size - input_width, 0)

            top_padded = height_padded // 2
            bottom_padded = height_padded - top_padded
            left_padded = width_padded // 2
            right_padded = width_padded - left_padded

            self.height_padded = height_padded
            self.width_padded = width_padded
            self.top_padded = top_padded
            self.bottom_padded = bottom_padded
            self.left_padded = left_padded
            self.right_padded = right_padded

            return np.pad(
                input_,
                ((0, 0), (top_padded, bottom_padded), (left_padded, right_padded), (0, 0)),
                mode="constant"
            )

        else:
            return input_

    @staticmethod
    @jit(nopython=True)
    def _forward(input_padded: np.array, filters: np.array, bias: np.array, stride: int) -> np.array:
        """
        Numba-jitted forward convolution
        Args:
            input_padded: (batch_size, hight, width, channels)
            filters: (num_filters, kernel_size, kernel_size, channels)
            stride: int
        Returns:
            output: (batch_size, hight_ouput, width_output, num_filters)
        """
        batch_size, height_input, width_input, channels = input_padded.shape
        num_filters, kernel_size, _, _ = filters.shape
        height_output = (height_input - kernel_size) // stride + 1
        width_output  = (width_input - kernel_size) // stride + 1

        output = np.zeros((batch_size, height_output, width_output, num_filters))

        for p in range(batch_size):
            for h in range(height_output):
                for w in range(width_output):
                    window = input_padded[p,
                                 h * stride: h * stride + kernel_size,
                                 w * stride: w * stride + kernel_size,
                                 :
                             ]

                    for f in range(num_filters):
                        filt = filters[f]
                        output[p, h, w, f] = np.sum(window * filt) + bias[f]

        return output

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

        grad_in_padded, grad_filters, grad_bias = self._backward(
            grad_out, self.last_input_padded, self.filters, self.stride
        )

        def clip_grad_norm(grad, max_norm):
            norm = np.linalg.norm(grad)
            if norm > max_norm:
                grad = grad * (max_norm / norm)
            return grad

        grad_filters = clip_grad_norm(grad_filters, 5.)
        grad_bias    = clip_grad_norm(grad_bias, 5.)

        self.filters -= learning_rate * grad_filters
        self.bias -= learning_rate * grad_bias

        if self.padding == "valid":
            grad_in = grad_in_padded
        elif self.padding == "same":
            grad_in = grad_in_padded[
                          :,
                          self.top_padded : grad_in_padded.shape[1] - self.bottom_padded,
                          self.left_padded : grad_in_padded.shape[2] - self.right_padded,
                          :
                      ]
        else:
            raise ValueError(f"Unsupported padding mode: {self.padding}")

        return grad_in

    @staticmethod
    @jit(nopython=True)
    def _backward(grad_out: np.array, input_padded: np.array, filters: np.array, stride: int):
        """
        Numba-jitted backward convolution
        Args:
            grad_out:    (batch_size, height_output, width_output, num_filters) gradient wrt output
            input_padded: (batch_size, height_input, width_input, channels) padded input
            filters:  (num_filters, kernel_size, kernel_size, channels)
            stride:   int
        Returns:
            grad_in:   (batch_size, in_h, in_w, channels) gradient wrt input
            grad_filters: (num_filters, kernel_size, kernel_size, channels) gradient wrt filters
        """
        batch_size, height_input, width_input, channels = input_padded.shape
        num_filters, kernel_size, _, _ = filters.shape
        _, height_output, width_output, _ = grad_out.shape

        grad_in = np.zeros((batch_size, height_input, width_input, channels))
        grad_filters = np.zeros((num_filters, kernel_size, kernel_size, channels))
        grad_bias = np.zeros(num_filters)

        for p in range(batch_size):
            for h in range(height_output):
                for w in range(width_output):
                    for f in range(num_filters):
                        grad_val = grad_out[p, h, w, f]

                        h_start = h * stride
                        w_start = w * stride
                        h_end = h_start + kernel_size
                        w_end = w_start + kernel_size

                        # Accumulate gradient wrt input
                        grad_in[p, h_start: h_end, w_start: w_end, :] += filters[f] * grad_val

                        # Accumulate gradient wrt filter
                        grad_filters[f] += input_padded[p, h_start: h_end, w_start: w_end, :] * grad_val

                        # Accumulate gradient wrt bias
                        grad_bias[f] += grad_val

        return grad_in, grad_filters, grad_bias


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
        windows_flattened = windows.reshape(batch_size, output_height, output_width, channels, -1)

        # Both max_indices and output have shape (batch_size, output_height, output_width, channels)
        self.max_indices = np.argmax(windows_flattened, axis=-1)
        output = np.max(windows_flattened, axis=-1)

        return output


    def backward(self, grad_out: np.array, learning_rate: float) -> np.array:
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
        # He initialization
        self.weights = np.random.randn(input_size, units) * np.sqrt(2 / input_size) # (input_size, units)
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

        self.bias -= learning_rate * grad_out.sum(axis=0) # (units,)
        self.weights -= learning_rate * self.last_input.T @ grad_out # (input_size, units)
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
        grad_in = self.last_output * (grad_out - dot)  # (batch_size, num_classes)
        return grad_in
        # return grad_out


class Conv2Dvec(Layer):
    def __init__(self, num_filters: int, kernel_size: int, stride: int, padding: str):
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

        if padding not in ["valid", "same"]:
            raise ValueError(f"Unsupported padding mode: {padding}. Please use 'valid' or 'same'.")
        self.padding = padding

    def _pad_input(self, input_: np.array) -> np.array:
        if self.padding == 'same':
            _, input_height, input_width, _ = input_.shape

            output_height = ceil(input_height / self.stride)
            output_width  = ceil(input_width / self.stride)

            padded_hight = max((output_height - 1) * self.stride + self.kernel_size - input_height, 0)
            padded_width = max((output_width - 1) * self.stride + self.kernel_size - input_width, 0)

            padded_top = padded_hight // 2
            padded_bottom = padded_hight - padded_top
            padded_left = padded_width // 2
            padded_right = padded_width - padded_left

            return np.pad(
                input_,
                ((0, 0), (padded_top, padded_bottom), (padded_left, padded_right), (0, 0)),
                mode="constant"
            )

        else:
            return input_

    def forward(self, input_: np.array):
        """
        Args:
            input_ (np.array): the input tensor with shape (batch_size, height, width, channels)
        """
        self.last_input = input_
        channels = input_.shape[-1]
        input_padded = self._pad_input(input_)

        if self.filters is None:
            # He-like initialization
            scale = np.sqrt(2.0 / (self.kernel_size * self.kernel_size * channels))
            self.filters = np.random.randn(self.num_filters, self.kernel_size, self.kernel_size, channels) * scale

        windows = sliding_window_view(input_padded, (self.kernel_size, self.kernel_size, channels))
        windows = windows[:, ::self.stride, ::self.stride, :, :, :]
        batches, output_height, output_width, _, _, _ = windows.shape

        cols = windows.reshape(batches * output_height * output_width, self.kernel_size * self.kernel_size * channels)
        filters_flattened = self.filters.reshape(self.num_filters, -1)

        output = cols @ filters_flattened.T
        output = output.reshape(batches, output_height, output_width, self.num_filters)

        self.windows = cols
        self.output_shape = (batches, output_height, output_width)

        return output

    def _pad_input(self, input_: np.array) -> np.array:
        if self.padding == 'same':
            _, input_height, input_width, _ = input_.shape

            output_height = ceil(input_height / self.stride)
            output_width  = ceil(input_width / self.stride)

            padded_hight = max((output_height - 1) * self.stride + self.kernel_size - input_height, 0)
            padded_width = max((output_width - 1) * self.stride + self.kernel_size - input_width, 0)

            padded_top = padded_hight // 2
            padded_bottom = padded_hight - padded_top
            padded_left = padded_width // 2
            padded_right = padded_width - padded_left

            return np.pad(
                input_,
                ((0, 0), (padded_top, padded_bottom), (padded_left, padded_right), (0, 0)),
                mode="constant"
            )

        else:
            return input_

    def backward(self, grad_out: np.array, learning_rate: float):
        """
        Args:
            grad_out (np.array): gradient from next layer of shape
                                 (batch_size, out_height, out_width, num_filters)
            learning_rate (float)
        Returns:
            grad_in (np.array): gradient w.r.t the input of shape same as last_input
        """
        batches, input_height, input_width, channels = self.last_input.shape
        _, output_height, output_width = self.output_shape

        grad_out_flattened = grad_out.reshape(batches * output_height * output_width, self.num_filters)

        # Gradient w.r.t filters
        grad_filters = grad_out_flattened.T @ self.windows
        grad_filters = grad_filters.reshape(self.filters.shape)

        # Gradient w.r.t input cols
        filters_flattened = self.filters.reshape(self.num_filters, -1)
        dcols = grad_out_flattened @ filters_flattened  # (batches * output_height * out_width, kernel_size * kernel_size * channels)

        dpatches = dcols.reshape(batches, output_height, output_width, self.kernel_size * self.kernel_size * channels)

        # Scatter patches back into input
        grad_in_padded = np.zeros_like(self._pad_input(self.last_input))

        for i in range(output_height):
            for j in range(output_width):
                grad_in_padded[
                    :,
                    i * self.stride: i * self.stride + self.kernel_size,
                    j * self.stride: j * self.stride + self.kernel_size,
                    :
                ] += dpatches[:, i, j].reshape(batches, self.kernel_size, self.kernel_size, channels)

        if self.padding == "valid":
            grad_in = grad_in_padded
        elif self.padding == "same":
            _, input_height, input_width, _ = self.last_input_.shape

            output_height = ceil(input_height / self.stride)
            output_width  = ceil(input_width / self.stride)

            padded_hight = max((output_height - 1) * self.stride + self.kernel_size - input_height, 0)
            padded_width = max((output_width - 1) * self.stride + self.kernel_size - input_width, 0)

            padded_top = padded_hight // 2
            padded_bottom = padded_hight - padded_top
            padded_left = padded_width // 2
            padded_right = padded_width - padded_left

            grad_in = grad_in_padded[
                          :,
                          padded_top : grad_in_padded.shape[1] - padded_bottom,
                          padded_left : grad_in_padded.shape[2] - padded_right,
                          :
                      ]
        else:
            raise ValueError(f"Unsupported padding mode: {self.padding}")

        self.filters -= learning_rate * grad_filters

        return grad_in







