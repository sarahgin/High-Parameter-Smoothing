import im_utils
import utils
import tensorflow as tf
import glob
import re
import numpy as np
import cv2

"""main"""
def main():
            model_folder = '.\\outputs\\D\\'
            model_output_folder = '.\\models-result\\D\\'
            with tf.Session() as sess:

                new_saver = tf.train.import_meta_graph(model_folder + 'my-test-model.meta')
                new_saver.restore(sess, tf.train.latest_checkpoint(model_folder))
                graph = tf.get_default_graph()

                tensors_per_node = [node.values() for node in graph.get_operations()]
                tensor_names = [tensor.name for tensors in tensors_per_node for tensor in tensors]
                print(tensor_names)

                for filename in glob.glob(model_output_folder + 'input\\' + '*.jpg'):
                    print(filename)
                    filenameOnly = re.search("\\d*\\.jpg$", filename).group()
                    filenameOnly = filenameOnly.replace('.jpg', '')

                    image = cv2.imread(filename)
                    image = np.float32(image)
                    grayImage = im_utils.toGray(image)
                    sobelimg = im_utils.get_sobel(grayImage)
                    utils.save_image(sobelimg, model_output_folder + 'sobel\\' + filenameOnly + '.jpg')

                    image = im_utils.add_one_dim(image)
                    sobelimg = im_utils.add_one_dim(sobelimg)

                    input = tf.get_default_graph().get_tensor_by_name("images_placeholder:0")
                    gt = tf.get_default_graph().get_tensor_by_name("gt_placeholder:0")
                    sobels = tf.get_default_graph().get_tensor_by_name("sobels_placeholder:0")

                    prediction = tf.get_default_graph().get_tensor_by_name("Prediction:0")

                    outputimg = sess.run(fetches=[prediction], feed_dict={input: image,
                                                                          sobels: sobelimg,
                                                                          gt: np.random.rand(sobelimg.shape[0], sobelimg.shape[1], sobelimg.shape[2])})
                    utils.save_image(255 * im_utils.normImage(outputimg[0].squeeze()), model_output_folder + 'output\\' + filenameOnly + '.jpg')
                return
        # end of model

if __name__ == '__main__':
    main()