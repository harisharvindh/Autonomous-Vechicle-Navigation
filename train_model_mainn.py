# -*- coding: utf-8 -*-
import pandas as pd
import cv2, os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.image as mpimg
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LeakyReLU
from keras.layers import MaxPooling2D, BatchNormalization, SpatialDropout2D


class PowerMode_autopilot:

    def __init__(self, data_path='data', learning_rate=1.0e-4, keep_prob=0.5, batch_size=40,
                 save_best_only=True, test_size=0.2, steps_per_epoch=20000, epochs=10):
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS = 66, 200, 3
        self.data_path = data_path
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.save_best_only = save_best_only
        self.batch_size = batch_size
        self.test_size = test_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs

    def load_data(self):
        """
        Read data from driving_log.csv file, then split the data into training set and validation set,
        For every piece of data,
        X represents images of left, center, right cameras
        y represents steering value, as reference
        :return: training sets and validation sets of images and steering value
        """
        data_df = pd.read_csv(os.path.join(os.getcwd(), self.data_path, 'driving_log.csv')
                              , names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
        X = data_df[['center', 'left', 'right']].values
        y = data_df['steering'].values
        # split the whole data into training and validation set RANDOMLY
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=self.test_size, random_state=0)
        return X_train, X_valid, y_train, y_valid

    def build_model(self):
        """
        build a model using keras.Sequential function
        :return: the model
        """
        model = Sequential()
        
        model.add(
            Lambda(lambda x: x / 127.5 - 1.0, input_shape=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS)))

        # five 2D convolution layers, 'elu' means Exponential Linear Unit function
        model.add(Conv2D(24, (5, 5), activation='elu',  strides=2, kernel_regularizer='l2'))
        #model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(self.keep_prob))

        model.add(Conv2D(36, (5, 5), activation='elu', strides=2, kernel_regularizer='l2'))
        #model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization())
        #model.add(SpatialDropout2D(self.keep_prob))

        model.add(Conv2D(48, (5,5), activation='elu', kernel_regularizer='l2'))
        #model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization())

        model.add(Conv2D(64,(3,3), activation='elu',padding='same', kernel_regularizer='l2'))
        #model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3,3), activation='elu',  padding= 'same', kernel_regularizer='l2'))
        #model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization())
        #model.add(SpatialDropout2D(self.keep_prob))

        model.add(Dropout(self.keep_prob))

        model.add(Flatten())

        model.add(Dense(128, activation='elu'))
        model.add(Dense(100, activation='elu', kernel_regularizer='l2'))
        model.add(Dense(50,activation='elu', kernel_regularizer='l2'))
        model.add(Dense(10,activation='elu', kernel_regularizer='l2'))
        model.add(Dense(1))

        model.summary()
        return model
        

                  

        

        


        #model.add(Conv2D(24, (5, 5), activation='elu', strides=2))
        #model.add(Conv2D(36, (5, 5), activation='elu', strides=2))
        #model.add(Conv2D(48, (5, 5), activation='elu', strides=2))
        #model.add(Conv2D(64, (3, 3), activation='elu'))
        #model.add(Conv2D(64, (3, 3), activation='elu'))

        # dropout layer, sets input units to 0 with a frequency of keep_prob var, to help prevent overfitting
        #model.add(Dropout(self.keep_prob))

        # three fully-connected layers and output layer after flattening
        #model.add(Flatten())
        #model.add(Dense(100, activation='elu'))
        #model.add(Dense(50, activation='elu'))
        #model.add(Dense(10, activation='elu'))
        #model.add(Dense(1))
        #model.summary()
        #return model

    def load_image(self, image_file):
        """
        read an image file
        :param image_file: image file name
        :return: the array representing the image
        """
        return mpimg.imread(os.path.join(self.data_path, image_file.strip()))

    """
    augment image
    """

    def augment(self, center, left, right, steering_angle, range_x=100, range_y=10):
        """
        image augmentation using following functions
        :param center:
        :param left:
        :param right:
        :param steering_angle:
        :param range_x:
        :param range_y:
        :return:
        """
        image, steering_angle = self.choose_image(center, left, right, steering_angle)
        image, steering_angle = self.random_flip(image, steering_angle)
        image, steering_angle = self.random_translate(image, steering_angle, range_x, range_y)
        image = self.random_shadow(image)
        image = self.random_brightness(image)

        return image, steering_angle

    def choose_image(self, center, left, right, steering_angle):
        """
        randomly choose an image: center, left, or right
        in case of left image or right image, steering angle needs to be adjusted accordingly
        :param center:
        :param left:
        :param right: images to choose from
        :param steering_angle:
        :return: result of choosing
        """
        choice = np.random.choice(3)
        if choice == 0:
            return self.load_image(left), steering_angle + 0.2
        elif choice == 1:
            return self.load_image(right), steering_angle - 0.2
        return self.load_image(center), steering_angle

    def random_flip(self, image, steering_angle):
        """
        randomly flip the image and invert steering angle accordingly or not,
        in case one direction steering happens way more than the other direction,
        which could lead to lack of training data diversity
        :param image:
        :param steering_angle:
        :return: image after flip
        """
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle
        return image, steering_angle

    def random_translate(self, image, steering_angle, range_x, range_y):
        """
        translate the image horizontally and vertically, randomly.
        modify the steering angle accordingly
        :param image:
        :param steering_angle:
        :param range_x:
        :param range_y:
        :return:
        """
        # getting a transformation matrix, trans_x and trans_y denote horizontal and vertical shift value
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        steering_angle += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])

        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
        return image, steering_angle

    def random_shadow(self, image):
        """
        add some random shadow because there are different light conditions in different situations
        :param image:
        :return:
        """
        # get the range of the shadow
        x1, y1 = self.IMAGE_WIDTH * np.random.rand(), 0
        x2, y2 = self.IMAGE_WIDTH * np.random.rand(), self.IMAGE_HEIGHT

        xm, ym = np.mgrid[0:image.shape[0], 0:image.shape[1]]

        mask = np.zeros_like(image[:, :, 1])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    def random_brightness(self, image):
        """
        adjust image brightness randomly
        convert the image to HSV model to adjust brightness more easily
        :param image:
        :return:
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:, :, 2] = hsv[:, :, 2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    """
    preprocess image
    """

    def preprocess(self, image):
        """
        preprocess the image, using the following functions
        :param image:
        :return:
        """
        image = self.crop(image)
        image = self.resize(image)
        image = self.rgb2yuv(image)
        return image

    def crop(self, image):
        """
        crop the image, keep the pixels between 60 from top and 25 from bottom,
        to get rid of the sky and the car itself, focus on the road
        :param image:
        :return:
        """
        return image[60:-25, :, :]

    def resize(self, image):
        """
        resize the image, using cv2.INTER_AREA as interpolation since we are shrinking the image
        :param image:
        :return:
        """
        return cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), cv2.INTER_AREA)

    def rgb2yuv(self, image):
        """
        convert the image from RGB format to YUV format
        :param image:
        :return:
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    """
    generate the training data (images and steering angles)
    """

    def batch_generator(self, image_paths, steering_angles, is_training):
        """
        generate a batch of training images and corresponding steering angles
        :param image_paths:
        :param steering_angles:
        :param is_training:
        :return:
        """
        # initialize a batch of images and steering angles with random values (faster than set all initial values to 0s)
        images = np.empty([self.batch_size, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS])
        steers = np.empty(self.batch_size)
        while True:
            i = 0
            for index in np.random.permutation(image_paths.shape[0]):
                center, left, right = image_paths[index]
                steering_angle = steering_angles[index]
                # enhance the images with probability of 0.6
                if is_training and np.random.rand() < 0.6:
                    image, steering_angle = self.augment(center, left, right, steering_angle)
                else:
                    image = self.load_image(center)
                # add the processed images and steering angle to the batch
                images[i] = self.preprocess(image)
                steers[i] = steering_angle
                i += 1
                # finish when the batch is filled
                if i == self.batch_size:
                    break
            yield images, steers

    """
    train the model
    """

    def train_model(self, model, X_train, X_valid, y_train, y_valid):
        """
        train the model
        :param model:
        :param X_train:
        :param X_valid:
        :param y_train:
        :param y_valid:
        :return:
        """
        # save model data after each epoch
        checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                     monitor='val_loss',
                                     verbose=0,
                                     # if save_vest_only is True, The model data with the best recent validation
                                     # error will be saved
                                     save_best_only=self.save_best_only,
                                     mode='auto')

        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

        # CPU does the real-time data enhancement of the image, and the model is trained in GPU in parallel
        model.fit_generator(self.batch_generator(X_train, y_train, True),
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs,
                            # max_queue_size=1,
                            validation_data=self.batch_generator(X_valid, y_valid, False),
                            validation_steps=len(X_valid),
                            callbacks=[checkpoint],
                            verbose=1)


def main():
    autopilot = PowerMode_autopilot(data_path='data', learning_rate=1.0e-4, keep_prob=0.5, batch_size=40,
                                    save_best_only=True, test_size=0.2, steps_per_epoch=2000, epochs=10)

    data = autopilot.load_data()

    model = autopilot.build_model()

    autopilot.train_model(model, *data)


if __name__ == '__main__':
    main()
