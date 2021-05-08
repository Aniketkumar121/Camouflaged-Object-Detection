from SINet import SINet_ResNet50

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkagg')

import numpy as np
from random import randrange
import argparse

import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.sm_losses = []
        self.im_losses = []
        

    def on_batch_end(self, batch, logs={}):
        if batch % 10 == 0:
            self.losses.append(logs.get('loss'))
            self.sm_losses.append(logs.get('SM_loss'))
            self.im_losses.append(logs.get('IM_loss'))

    def plot(self):

        f = plt.figure()
        f.set_figwidth(15)
        f.set_figheight(4)


        plt.subplot(1, 3, 1)
        l = loss_history.losses
        plt.xlabel("Batches")
        plt.ylabel('Total loss')
        plt.plot(self.losses)

        plt.subplot(1, 3, 2)
        plt.xlabel("Batches")
        plt.ylabel('SM loss')
        plt.plot(self.sm_losses)


        plt.subplot(1, 3, 3)
        plt.xlabel("Batches")
        plt.ylabel('IM loss')
        plt.plot(self.im_losses)

        plt.tight_layout()
        plt.show()



def train(args):

    datagen = ImageDataGenerator(rescale=1./255)

    # Means import all the dataset in one batch ( will have batches during the training )
    hyper_batch_size = 6000

    train_images_generator = datagen.flow_from_directory(args.images, class_mode=None, target_size=(352, 352), batch_size=hyper_batch_size, shuffle=False)
    train_labels_generator = datagen.flow_from_directory(args.labels, class_mode=None, target_size=(352, 352), batch_size=hyper_batch_size, shuffle=False)    

    print('Loading x')
    X = train_images_generator[0]
    print('Loading y')
    Y = train_labels_generator[0]

    shuffler = np.random.permutation(len(X))

    X = X[shuffler]
    Y = Y[shuffler]

    thresh = hyper_batch_size*3//4

    X_train, y_train = X[:thresh], Y[:thresh]
    X_val, y_val = X[thresh:], Y[thresh:]


    if args.save and (args.check_path is not None):
        model_checkpoint_callback = ModelCheckpoint(
        filepath=args.check_path,
        save_weights_only=True)
        loss_history = LossHistory()

    opt = optimizers.Adam(learning_rate=1e-4)

    model = SINet_ResNet50()

    model.compile(optimizer=opt, loss=["binary_crossentropy", "binary_crossentropy"])

    if args.save:
        h = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_val, y_val), callbacks=[model_checkpoint_callback, loss_history])

    else:
        h = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_val, y_val))

    model.save(args.save_path)

    if args.visualize:
        print('Visualizing losses:')
        loss_history.plot()

    if args.test:
        k = randrange(len(X_val))

        _, cam = model.predict(np.array([X[k]]))

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
        plt.imshow(y[k])

        plt.tight_layout()



def main():
    
    parser = argparse.ArgumentParser(description='Arguments for training the network.')

    parser.add_argument('--images', type=str, required=True,
                    help='Path to the images')  

    parser.add_argument('--labels', type=str, required=True,
                    help='Path to the labels')  

    parser.add_argument('--epochs', type=int, required=False, default=10,
                    help="Number of epochs")

    parser.add_argument('--batch-size', type=int, required=False, default=12,
                    help="Batch size")

    parser.add_argument('--save', type=int, required=False, default=0,
                    help='1 for saving at every checkpoint')  

    parser.add_argument('--check-path', type=str, required=False,
                    help='Path for saving the checkpoints')  

    parser.add_argument('--visualize', type=int, required=False, default=0,
                    help='1 for visualizing the losses at the end of the training')  

    parser.add_argument('--save-path', type=str, required=False, default='/',
                    help='Path for saving the model at the end')  

    args = parser.parse_args()

    if args.save and (args.check_path is None):
        parser.error("--save requires --check-path.")

    train(args)


if __name__ == "__main__":
    main()