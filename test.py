import numpy as np

if __name__ == '__main__':
    #print(np.random.rand(((16,16), (16,), (16,64), (64,), (64,4), (4,))))
    first_layer = np.random.rand(16,16)
    second_layer = np.random.rand(16,)
    third_layer = np.random.rand(16,64)
    fourth_layer = np.random.rand(64,)
    fifth_layer = np.random.rand(64,4)
    sixth_layer = np.random.rand(4,)

    print(np.array((first_layer, second_layer, third_layer, fourth_layer, fifth_layer, sixth_layer)))    