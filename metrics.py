# import tensorflow as tf
#
#
# def f1(predictions, labels):
#     numerator = tf.reduce_sum(labels * predictions, axis=[0, 1, 2])
#     denominator = tf.reduce_sum(labels, axis=[0, 1, 2]) + tf.reduce_sum(predictions, axis=[0, 1, 2])
#     dice = (2. * numerator + 1.) / (denominator + 1.)
#     return tf.compat.v1.metrics.mean(dice)
