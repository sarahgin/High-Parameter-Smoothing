import argparse
import tensorflow as tf
import bsd_net
import vgg19

"""parsing and configuration"""


def parse_args():
    desc = "TensorFlow implementation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--train_input', type=str, default='INPUT MISSING',
                        help='Path to training images', required=False)
    parser.add_argument('--train_gt', type=str, default='INPUT MISSING',
                        help='Path to training ground truth', required=False)
    parser.add_argument('--train_sobel', type=str, default='INPUT MISSING',
                        help='Path to training original gradients', required=False)
    parser.add_argument('--output_folder', type=str, default='INPUT MISSING',
                        help='Output folder for model, select testing images, and plots', required=False)

    parser.add_argument('--model_path', type=str, default='pre_trained_model',
                        help='The directory where the pre-trained model was saved')
    parser.add_argument('--content_loss_norm_type', type=int, default=3, choices=[1, 2, 3],
                        help='Different types of normalization for content loss')
    return parser.parse_args()


"""main"""


def main():
    # parse arguments
    args = parse_args()

    # initiate VGG19 model
    model_file_path = args.model_path + '/' + vgg19.MODEL_FILE_NAME
    vgg_net = vgg19.VGG19(model_file_path)

    # open session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # build the graph
    st = bsd_net.BSD_net(session=sess,
                         training_images_path=args.train_input,
                         training_gt_path=args.train_gt,
                         training_sobel_path=args.train_sobel,
                         output_folder=args.output_folder,
                         net=vgg_net,
                         content_loss_norm_type=args.content_loss_norm_type)

    # launch the graph in a session
    st.runme()

    # close session
    sess.close()
    print('current session is finished ok')


if __name__ == '__main__':
    main()
