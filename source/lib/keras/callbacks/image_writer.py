from keras.callbacks import Callback
from lib.keras.tensorflow_utils.tensorboard_manager import TensorBoardManager as TBManager
import tensorflow as tf
import numpy as np
from lib.thread_pool_manager import ThreadPoolManager


class ImageWriter(Callback):
    """
    This class is used as keras callback for writing custom images on tensorboard during the training phase.
    The writing part is handle asynchronously in order to speed up the inter epoch phase
    """
    def __init__(self, data: tuple = (None, None), max_imgs=5, name='images', freq=1):
        """
        :param data: Is a tuple with the images to print. The first component is a list of input images.
            The second one is a list of target images. The images should have all the same dimension in order
            to fit the pre-allocated tensors
        :param max_imgs: is the maximum number if images that the callback will write on tensorboard
        :param name: is used to recognise them in tensorboard. the default name in 'images'
        :param freq: is the writing frequency. every freq epochs the model is tested with the input images
            and the results are written in the tensorboard log file.
        """
        self.tb_manager = TBManager(scope_name=name)
        super(ImageWriter, self).__init__()
        # Preparing the data
        self.input_images = data[0][0:max_imgs]
        self.target_images = data[1][0:max_imgs]
        self.input_images_3d = None
        self.name = name
        self.freq = freq
        self.max_imgs = max_imgs
        # Preparing the tensors for the images
        self.image_tensor = None
        self.output_tensor = None
        self.target_tensor = None
        # Getting the global thread pool manager
        self.pool = ThreadPoolManager.get_thread_pool()

    def on_train_begin(self, logs=None):
        if self.input_images is not None and self.target_images is not None:
            self.input_images_3d = self.input_images[:, :, :, 0:3]

            # Creating the tensors
            self.image_tensor = tf.placeholder(dtype=tf.float32,
                                               shape=np.shape(self.input_images_3d),
                                               name=self.name + "_X")
            self.output_tensor = tf.placeholder(dtype=tf.float32,
                                                shape=np.shape(self.model.predict(self.input_images)),
                                                name=self.name + "_Y")
            self.target_tensor = tf.placeholder(dtype=tf.float32,
                                                shape=np.shape(self.target_images),
                                                name=self.name + "_T")
            # Adding the image tensors to the tensorbiard manager
            self.tb_manager.add_images(self.image_tensor, name=self.name + "_X", max_out=self.max_imgs)
            self.tb_manager.add_images(self.output_tensor, name=self.name + "_Y", max_out=self.max_imgs)
            self.tb_manager.add_images(self.target_tensor, name=self.name + "_T", max_out=self.max_imgs)

    def on_epoch_end(self, epoch, logs=None):
        """
        On the end of the epoch the model is tested and the images are written to the log file.
        This function is called by the fit function of the model; it should never been called by the programmer
        :param epoch: is the current epoch
        :param logs: dictionary used by the fit function
        """
        logs = logs or {}
        if epoch % self.freq == 0:
            if self.input_images is not None and self.target_images is not None:
                output_img = self.model.predict(self.input_images)
                self.pool.submit(self.__write_step, output_img, epoch, tf.get_default_graph())

    def __write_step(self, heat_maps, epoch, cur_graph):
        with tf.Session(graph=cur_graph) as s:
            summary = s.run(self.tb_manager.get_runnable(),
                            feed_dict={self.image_tensor: self.input_images_3d,
                                       self.output_tensor: heat_maps,
                                       self.target_tensor: self.target_images})[0]
            self.tb_manager.write_step(summary, epoch)
