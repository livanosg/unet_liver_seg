import tensorflow as tf


class Losses:
    def __init__(self, logits, labels):
        self.logits = logits
        self.labels = labels

    def weighted_crossentropy(self):  # todo fix weights shape
        """weighted softmax_cross_entropy"""
        with tf.name_scope('Weighted_Crossentropy'):
            class_freq = tf.reduce_sum(self.labels, axis=[0, 1, 2], keepdims=True)
            class_freq = tf.math.maximum(class_freq, 1)
            weights = tf.math.pow(tf.math.divide(tf.reduce_sum(class_freq), class_freq), 0.5)
            weights = tf.multiply(weights, self.labels)
            weights = tf.reduce_max(weights, -1)
            loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=self.labels, logits=self.logits,
                                                             weights=weights)
        return loss

    def log_dice_loss(self):
        """both tensors are [b, h, w, classes] and y_pred is in probs form"""
        with tf.name_scope('Weighted_Generalized_Dice_Log_Loss'):
            predictions = tf.math.softmax(self.logits, -1)
            class_freq = tf.reduce_sum(self.labels, axis=[0, 1, 2])
            class_freq = tf.math.maximum(class_freq, 1)
            weights = 1 / (class_freq ** 2)

            numerator = tf.reduce_sum(self.labels * predictions, axis=[0, 1, 2])
            denominator = tf.reduce_sum(self.labels + predictions, axis=[0, 1, 2])
            dice = (2 * weights * (numerator + 1)) / (weights * (denominator + 1))
            loss = tf.math.reduce_mean(- tf.math.log(dice))
        return loss

    def custom_loss(self):
        with tf.name_scope('Custom_loss'):
            dice_loss = self.log_dice_loss()
            wce_loss = self.weighted_crossentropy()
            loss = tf.math.multiply(.3, dice_loss) + tf.math.multiply(0.7, wce_loss)
        return loss
