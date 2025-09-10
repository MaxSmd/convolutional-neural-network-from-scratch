from typing import Callable
import numpy as np

class Sequential:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            loss: Callable[[np.ndarray, np.ndarray], np.ndarray],
            loss_prime: Callable[[np.ndarray, np.ndarray], np.ndarray],
            epochs: int = 100,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            verbose: bool = True,
    ):
        """
        Args:
            x (np.ndarray): (num_samples, height, width, channels)
            y (np.array): (num_samples, unique(samples))
            loss (Callable[[np.ndarray, np.ndarray], np.ndarray]): loss function
            loss_prime (Callable[[np.ndarray, np.ndarray], np.ndarray]): derivation of the loss function
            epochs (int): number of epochs
            learning_rate (float): learning rate
            batch_size (int): batch size
            verbose (boolean): output epoch loss yes or no
        """
        n_train = x.shape[0]

        for epoch in range(epochs):
            perm = np.random.permutation(n_train)
            x_shuff, y_shuff = x[perm], y[perm]

            epoch_loss = 0.0
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                xb, yb = x_shuff[start:end], y_shuff[start:end]

                y_pred = self.predict(xb)

                batch_loss = loss(yb, y_pred)
                grad = loss_prime(yb, y_pred)

                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

                epoch_loss += batch_loss * len(xb)

            epoch_loss /= n_train
            if verbose:
                print(f"{epoch + 1}/{epochs}, loss={epoch_loss:.6f}")





