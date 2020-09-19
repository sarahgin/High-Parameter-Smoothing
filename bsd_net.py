import matplotlib
import tensorflow as tf
import numpy as np
import logging
import os
import im_utils
import utils

matplotlib.use('Agg')

patch_stride = 50
patch_size = 100
batch_size = 20

full_image_rows = 481
full_image_cols = 321

numberOfTestImages = 3
numOfEpochs = 1


class BSD_net:

    def __init__(self,
                 training_images_path,
                 training_gt_path,
                 training_sobel_path,
                 output_folder,
                 session,
                 net,
                 content_loss_norm_type):

        self.train_input = training_images_path
        self.train_gt = training_gt_path
        self.train_sobel = training_sobel_path
        self.output_folder = output_folder

        self.test_input_path = training_images_path.replace('train', 'test')
        self.test_ground_truth_path = training_gt_path.replace('train', 'test')
        self.test_sobel_path = training_sobel_path.replace('train', 'test')

        self.net = net
        self.sess = session
        self.content_loss_norm_type = content_loss_norm_type
        logging.basicConfig(filename=self.output_folder + 'mylogfile.log', level=logging.DEBUG)

    def _build_graph(self, inputImages, sobelImages):

        # get all 16 layers - these are fixed and upsampled, returned from vgg19 feed_forward
        activations_all_layers = self.net.feed_forward(inputImages, scope="myscope")

        # TO THREE CHANNELS AND UPSAMPLE ALL
        tensors = []
        for k in range(16):
            logging.info(str(k))
            currLayer = activations_all_layers[k]
            numOfChannels = currLayer.shape[3]

            input_rows = tf.shape(inputImages)[1]
            input_cols = tf.shape(inputImages)[2]

            # upsample the block to the full input image size (patch in train or full in test)
            # currLayer = tf.image.resize_images(images=currLayer, size=[patch_size, patch_size])

            # bilinear interpolation
            currLayer = tf.image.resize_images(images=currLayer, size=[input_rows, input_cols])

            # these variables are used to collapse each layer to 3-channels only
            num_of_layers_to = 3
            W_conv1 = weight_variable([1, 1, int(numOfChannels), num_of_layers_to])  # COLLAPSE LAYERS
            b_conv1 = bias_variable([num_of_layers_to])  # one for each filter
            h_conv1 = conv2d(currLayer, W_conv1) + b_conv1  # collapsed
            h_conv1 = tf.nn.relu(h_conv1)
            tensors.append(h_conv1)

        # stack all collapsed layers (16 x 3 = 48)
        h_stack = tf.concat([tensors[0],
                             # tensors[1]], axis=3)
                             tensors[2],
                             # tensors[3]], axis=3)
                             tensors[4],  # ], axis=3)
                             # tensors[5],
                             # tensors[6],
                             # tensors[7],  # axis=3)
                             tensors[8],
                             # tensors[9],
                             # tensors[10],
                             # tensors[11],  # axis=3)
                             tensors[12]], axis=3)
        # tensors[13],
        # tensors[14],
        # tensors[15]], axis=3)

        # collapse the stack into a single channel
        numOfChannels = h_stack.get_shape()[3]

        # sobel is grayscale
        # the concat stack to 48 --> 64 channels
        W_conv2 = weight_variable([1, 1, int(numOfChannels), 64])
        b_conv2 = bias_variable([64])
        h_conv2 = conv2d(h_stack, W_conv2) + b_conv2

        # finally conv 1x1 so that we will have ONE channel only
        W_conv3 = weight_variable([1, 1, 64, 1])
        b_conv3 = bias_variable([1])

        # sobel is rgb
        # W_conv2 = weight_variable([1, 1, int(numOfChannels), 3])
        # b_conv2 = bias_variable([3])

        h_conv3 = conv2d(h_conv2, W_conv3) + b_conv3
        h_conv3 = tf.squeeze(h_conv3)

        # element-wise multiplication of stack with sobel
        # mult_h = tf.multiply(h_conv2, sobelImages)

        # go directly to ground truth
        mult_h = h_conv3

        # NORMALIZATION
        # OPTION 1
        # mult_h = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), mult_h)
        # OPTION 2
        # mult_h = tf.nn.sigmoid(mult_h)
        # mult_h = tf.nn.tanh(mult_h)
        # mult_h = tf.nn.relu(mult_h)
        # OPTION 3
        mult_h = tf.div(
            tf.subtract(
                mult_h,
                tf.reduce_min(mult_h)
            ),
            tf.subtract(
                tf.reduce_max(mult_h),
                tf.reduce_min(mult_h)
            )
            , name="Prediction")

        return mult_h

    def runme(self):
        # load data
        self.full_images = im_utils.loadImages(self.train_input, full_image_rows, full_image_cols, patch_size,
                                               patch_stride, False)
        logging.info('Loaded full-size images: ' + str(len(self.full_images)))
        self.full_ground_truth_images = im_utils.loadGroundTruth(self.train_gt, full_image_rows, full_image_cols,
                                                                 patch_size, patch_stride, False)
        logging.info('Loaded full-size ground truths: ' + str(len(self.full_ground_truth_images)))
        self.full_sobel_images = im_utils.loadSobels(self.train_sobel, full_image_rows, full_image_cols, patch_size,
                                                     patch_stride, False)
        logging.info('Loaded full-size sobels: ' + str(len(self.full_sobel_images)))

        self.images = im_utils.loadImages(self.train_input, full_image_rows, full_image_cols, patch_size, patch_stride,
                                          True)
        logging.info('Loaded images: ' + str(len(self.images)))
        self.ground_truth_images = im_utils.loadGroundTruth(self.train_gt, full_image_rows, full_image_cols, patch_size,
                                                            patch_stride, True)
        logging.info('Loaded ground truths: ' + str(len(self.ground_truth_images)))
        self.sobel_images = im_utils.loadSobels(self.train_sobel, full_image_rows, full_image_cols, patch_size,
                                                patch_stride, True)
        logging.info('Loaded sobels: ' + str(len(self.sobel_images)))

        # load test data
        self.test_images = im_utils.loadImages(self.test_input_path, full_image_rows, full_image_cols, patch_size,
                                               patch_stride, False)
        logging.info('Loaded test images: ' + str(len(self.test_images)))
        self.test_ground_truth_images = im_utils.loadGroundTruth(self.test_ground_truth_path, full_image_rows,
                                                                 full_image_cols, patch_size, patch_stride, False)
        logging.info('Loaded test ground truths: ' + str(len(self.test_ground_truth_images)))
        self.test_sobel_images = im_utils.loadSobels(self.test_sobel_path, full_image_rows, full_image_cols, patch_size,
                                                     patch_stride, False)
        logging.info('Loaded test sobels: ' + str(len(self.test_sobel_images)))

        # prepare input placeholders
        batch_data = tf.placeholder(tf.float32, [None, None, None, 3], name="images_placeholder")
        batch_sobels = tf.placeholder(tf.float32, [None, None, None], name="sobels_placeholder")

        # prepare output placeholder
        batch_labels = tf.placeholder(tf.float32, [None, None, None], name="gt_placeholder")

        # build graph
        curr_im = self._build_graph(batch_data, batch_sobels)

        # main loss: element-wise subtract
        subtraction_result = tf.subtract(curr_im, batch_labels)
        loss_all_channels = tf.nn.l2_loss(subtraction_result)
        loss = tf.reduce_mean(loss_all_channels)

        # regularization loss: zero patches
        # batch_labels_images_sum = tf.reduce_sum(batch_labels, axis=[1,2])
        # zero_patches = tf.where(tf.equal(batch_labels_images_sum, 0))
        # loss_regularization = tf.nn.l2_loss(curr_im_gt_patches)
        # loss_regularization = tf.reduce_mean(loss_regularization)
        # loss = loss + loss_regularization

        # define loss and optimizer
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        if True:
            epoch = 1
            f = matplotlib.pyplot.figure(1)
            g = matplotlib.pyplot.figure(2)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                iteration = 1
                while epoch <= numOfEpochs:
                    logging.info('EPOCH ' + str(epoch))
                    indices = im_utils.next_epoch(batch_size, self.images)

                    for l in range(len(indices)):
                        current_indices = indices[l]
                        batch = im_utils.next_batch(current_indices,
                                                    self.images,
                                                    self.ground_truth_images,
                                                    self.sobel_images)

                        # evaluate a few training images
                        train_res_loss = loss.eval(feed_dict={batch_data: self.full_images[0:numberOfTestImages],
                                                              batch_labels: self.full_ground_truth_images[
                                                                            0:numberOfTestImages],
                                                              batch_sobels: self.full_sobel_images[
                                                                            0:numberOfTestImages]})
                        train_res_curr_im = curr_im.eval(feed_dict={batch_data: self.full_images[0:numberOfTestImages],
                                                                    batch_labels: self.full_ground_truth_images[
                                                                                  0:numberOfTestImages],
                                                                    batch_sobels: self.full_sobel_images[
                                                                                  0:numberOfTestImages]})
                        logging.info('iteration: ' + str(iteration)
                                     + ' train curr_im_shape: ' + str(train_res_curr_im.shape)
                                     + ' values: [' + str(np.amin(train_res_curr_im)) + ',' + str(
                            np.amax(train_res_curr_im)) + ']')

                        # evaluate a few testing images
                        test_res_loss = loss.eval(feed_dict={batch_data: self.test_images[0:numberOfTestImages],
                                                             batch_labels: self.test_ground_truth_images[
                                                                           0:numberOfTestImages],
                                                             batch_sobels: self.test_sobel_images[
                                                                           0:numberOfTestImages]})
                        test_res_curr_im = curr_im.eval(feed_dict={batch_data: self.test_images[0:numberOfTestImages],
                                                                   batch_labels: self.test_ground_truth_images[
                                                                                 0:numberOfTestImages],
                                                                   batch_sobels: self.test_sobel_images[
                                                                                 0:numberOfTestImages]})
                        logging.info('iteration: ' + str(iteration)
                                     + ' test curr_im_shape: ' + str(test_res_curr_im.shape)
                                     + ' values: [' + str(np.amin(test_res_curr_im)) + ',' + str(
                            np.amax(test_res_curr_im)) + ']')

                        # training step
                        train_step.run(feed_dict={batch_data: batch[0], batch_labels: batch[1], batch_sobels: batch[2]})

                        # save output images - 3 from train, 3 from test
                        for t in range(numberOfTestImages):
                            test_image = test_res_curr_im[t, :, :]
                            test_image = (test_image - np.amin(test_image)) / (
                                        np.amax(test_image) - np.amin(test_image))
                            test_image_folder = self.output_folder + 'test_image_' + str(t) + '/'
                            if not os.path.isdir(test_image_folder):
                                os.mkdir(test_image_folder)
                            if iteration % 100 == 0:
                                utils.save_image(255 * test_image, test_image_folder + str(iteration) + '.jpg')

                        for t in range(numberOfTestImages):
                            train_image = train_res_curr_im[t, :, :]
                            train_image = (train_image - np.amin(train_image)) / (
                                        np.amax(train_image) - np.amin(train_image))
                            train_image_folder = self.output_folder + 'train_image_' + str(t) + '/'
                            if not os.path.isdir(train_image_folder):
                                os.mkdir(train_image_folder)
                            if iteration % 100 == 0:
                                utils.save_image(255 * train_image, train_image_folder + str(iteration) + '.jpg')

                        # save loss progression plots
                        try:
                            matplotlib.pyplot.figure(1)
                            trainHandle, = matplotlib.pyplot.plot(iteration, train_res_loss, 'bo-')
                            matplotlib.pyplot.title('Learning progression')
                            matplotlib.pyplot.xlabel('Steps')
                            matplotlib.pyplot.ylabel('Loss value')
                            matplotlib.pyplot.legend([trainHandle], ['Training'])
                            f.savefig(self.output_folder + 'loss_training.png')

                            matplotlib.pyplot.figure(2)
                            testHandle, = matplotlib.pyplot.plot(iteration, test_res_loss, 'ro-')
                            matplotlib.pyplot.title('Learning progression')
                            matplotlib.pyplot.xlabel('Steps')
                            matplotlib.pyplot.ylabel('Loss value')
                            matplotlib.pyplot.legend([testHandle], ['Testing'])
                            g.savefig(self.output_folder + 'loss_testing.png')
                        except:
                            logging.info('EXCEPTION: ' + str(iteration))

                            logging.info('-----------------END OF ITERATION----------------')
                        iteration = iteration + 1

                    epoch = epoch + 1

                # """ get final result """
                saver = tf.train.Saver()
                saver.save(sess, self.output_folder + './my-test-model')

        # return loss
        return

    def _gram_matrix(self, tensor):

        shape = tensor.get_shape()

        # Get the number of feature channels for the input tensor,
        # which is assumed to be from a convolutional layer with 4-dim.
        num_channels = int(shape[3])

        # Reshape the tensor so it is a 2-dim matrix. This essentially
        # flattens the contents of each feature-channel.
        matrix = tf.reshape(tensor, shape=[-1, num_channels])

        # Calculate the Gram-matrix as the matrix-product of
        # the 2-dim matrix with itself. This calculates the
        # dot-products of all combinations of the feature-channels.
        gram = tf.matmul(tf.transpose(matrix), matrix)

        return gram


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.333, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
