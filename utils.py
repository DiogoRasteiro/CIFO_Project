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

    print(len(flat_weights))

    weights[0] = np.reshape(flat_weights[0:256], (16,16))
    weights[1] = np.reshape(flat_weights[256:271], (16,))
    weights[2] = np.reshape(flat_weights[272:1295], (16, 64))
    weights[3] = np.reshape(flat_weights[1296:1359], (64,))
    weights[4] = np.reshape(flat_weights[1360:1615], (64,4))
    weights[5] = np.reshape(flat_weights[1616:1619], (4,))

    return weights

player = models.Sequential()
player.add(layers.Dense(16, input_dim=16,activation='relu'))
player.add(layers.Dense(64, activation='relu'))
player.add(layers.Dense(4, activation='softmax'))

weights = player.get_weights()

print("\n MOMENT OF TRUTH\n")

print(weights == unflatten(flatten(weights)))

# np.concatenate((weights[0].flatten(),weights[1].flatten()))
