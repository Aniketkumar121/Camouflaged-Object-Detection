import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import preprocessing

class Dataset():
    def __init__(self, image_path='../tst/jpg/', label_path='../tst/jpg/', train_size=2 ):

        # Set the size of the training data 
        self.train_size = train_size

        # Import the images of format jpg
        self.images = [image_path + image for image in os.listdir(image_path) if image.endswith('.jpg')]

        # Import the labels of format jpg and png
        self.labels = [label_path + label for label in os.listdir(label_path) if label.endswith('.jpg') or label.endswith('.png')]

        # Sort the images and the labels
        self.images = sorted(self.images)
        self.labels = sorted(self.labels)

        # Filtering the files
        self.filter_files()

        print(len(self.images))
        print(len(self.labels))
        self.size = len(self.images)
        self.index = 0

    def __get_item__(self, index):
        image = self.rgb_loader(self.images[index])
        label = self.binary_loader(self.labels[index])
        image = self.image_transform(image)
        label = self.label_transform(label)
        return image, label
    
    def filter_files(self):
        """ Creating an array containing all the images and another containing all the labels
            Each image and its corresponding label have the same index in the two arrays."""

        assert len(self.images) == len(self.labels)
        images = []
        labels = []
        for img_path, label_path in zip(self.images, self.labels):

            img = Image.open(img_path)
            label = Image.open(label_path)

            if img.size == label.size:
                images.append(img)
                labels.append(label)

        self.images = images
        self.labels = labels

    def image_transform(self, img):
        img = preprocessing.image.img_to_array(img)
        img = tf.image.resize(img, (train_size, train_size))
        img = tf.keras.applications.resnet.preprocess_input(img)
        img = tf.reshape(img, [1, 3, train_size, train_size])
        return img

    def label_transform(self, label):
        label = preprocessing.image.img_to_array(label)
        label = tf.keras.applications.resnet.preprocess_input(label)
        return label

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        label = self.binary_loader(self.labels[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, label, name
