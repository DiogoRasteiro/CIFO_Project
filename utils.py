from matplotlib import pyplot as plt
from keras import layers, models
import numpy as np

def plot_c(c, alpha, threshold):
    p = [c]
    while c > threshold:
        c = c*alpha
        p.append(c)
    plt.plot(p)
    plt.show()
    
def flatten(weights):
    """Flattens the weight list provided by keras to a one-dimensional format

    Args:
        weights: a list of multi-dimensional numpy arrays representing weights in a NN

    Returns:
        flat_weights: a one-dimensional list of weights

    """
    flat_weights = np.concatenate((
        weights[0].flatten(),
        weights[1].flatten(),
        weights[2].flatten(),
        weights[3].flatten(),
        weights[4].flatten(),
        weights[5].flatten(),
        ))

    return flat_weights

def unflatten(flat_weights):
    """Transforms a one-dimensional list into a multi-dimensional object to be loaded into a Keras' NN

    Args:
        flat_weights: a one-dimensional list of weights

    Returns:
        weights: a list of multi-dimensional numpy arrays representing weights in a NN
    """
    weights = []

    #curr_weights = flat_weights[0:255]
    weights.append(np.zeros((16,16)))
    weights[0] = np.reshape(flat_weights[0:256], (16,16))
    weights.append(np.zeros((16,)))
    weights[1] = np.reshape(flat_weights[256:272], (16,))
    weights.append(np.zeros((16,64)))
    weights[2] = np.reshape(flat_weights[272:1296], (16, 64))
    weights.append(np.zeros((64,)))
    weights[3] = np.reshape(flat_weights[1296:1360], (64,))
    weights.append(np.zeros((64,4)))
    weights[4] = np.reshape(flat_weights[1360:1616], (64,4))
    weights.append(np.zeros((4,)))
    weights[5] = np.reshape(flat_weights[1616:1620], (4,))

    return weights

