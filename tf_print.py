"""
The default tf.Print op goes to STDERR
Use the function below to direct the output to stdout instead
Usage:
> x=tf.ones([1, 2])
> y=tf.zeros([1, 3])
> p = x*x
> p = tf_print(p, [x, y], "hello")
> p.eval()
hello [[ 0.  0.]]
hello [[ 1.  1.]]
"""
import sys
import tensorflow as tf
def tf_print(op, tensors, message=None):
    def print_message(x):
        sys.stdout.write(message + " %s\n" % x)
        return x

    prints = [tf.py_func(print_message, [tensor], tensor.dtype) for tensor in tensors]
    with tf.control_dependencies(prints):
        op = tf.identity(op)
    return op