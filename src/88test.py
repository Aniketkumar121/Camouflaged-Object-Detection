from SINet import SINet_ResNet50

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkagg')

from tensorflow.keras.layers import UpSampling2D, Activation
from tensorflow.keras.activations import sigmoid


def test(args):
    model = SINet_ResNet50()
    model.load_weights(args.model)


    image = ...
    label = ...

    _, cam = model.predict(image)

    cam = UpSampling2D(size=3, interpolation='bilinear')(cam)
    cam = Activation(sigmoid)(cam)
    cam = cam.numpy().squeeze()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(4)

    plt.subplot(1, 3, 1)
    plt.imshow(x[k])

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.imshow(label)

    plt.tight_layout()


def main():
    args.model 