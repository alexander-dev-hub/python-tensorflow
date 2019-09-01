import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import cifa_cnn

# data directory
RESULT_PATH = os.getcwd() + "/result"
#RESULT_PATH = "F:/tmp/cifar10_data/cifar-10-batches-py"

imageSize = 32
labelSize = 1
imageDepth = 3
debugEncodedImage = True

def unpickle(file):
    with open(os.path.join(RESULT_PATH, file), 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
        
    return dict

def display_cifar(images, size, all=False):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()

    if not all:
        im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
                    for i in range(size)])
    else:
        im = np.vstack([np.hstack([images[i] for i in range(size)])
                    for i in range(size)])
    plt.imshow(im)
    plt.show()

srcfiles=["train"]
#srcfiles=["test_batch"]
data = [unpickle(f) for f in srcfiles]
images = np.vstack([d["data"] for d in data])
n = len(images)
images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255
display_cifar(images, 10)


srcfiles=["test"]
#srcfiles=["test_batch"]
data = [unpickle(f) for f in srcfiles]
images = np.vstack([d["data"] for d in data])
n = len(images)
images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255
display_cifar(images, 2, True)