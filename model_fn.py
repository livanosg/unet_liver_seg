import tensorflow as tf
from tensorflow.compat.v1.train import AdamOptimizer as Adam
from tensorflow.python import summary
from losses import Losses
from tensorflow.compat.v1 import estimator
from model import Unet
from metrics import f1


def unet_model_fn(features, labels, mode, params):
    tf.local_variables_initializer()
    loss, train_op, = None, None
    eval_metric_ops, training_hooks, evaluation_hooks = None, None, None
    predictions_dict = None
    unet = Unet(params=params)
    logits = unet.model(input_tensor=features['image'])
    y_pred = tf.math.softmax(logits, axis=-1)
    output_img = tf.expand_dims(tf.cast(tf.math.argmax(y_pred, axis=-1) * 255, dtype=tf.uint8), axis=-1)

    if mode in (estimator.ModeKeys.TRAIN, estimator.ModeKeys.EVAL):

        with tf.name_scope('Loss_Calculation'):
            loss = Losses(logits=logits, labels=labels['label'])
            loss = loss.custom_loss()

        with tf.name_scope('Dice_Score_Calculation'):
            dice = f1(labels=labels['label'], predictions=y_pred)

        with tf.name_scope('Images_{}'.format(mode)):
            with tf.name_scope('Reformat_Outputs'):
                label = tf.expand_dims(tf.cast(tf.argmax(labels['label'], -1) * 255, dtype=tf.uint8), axis=-1)
                image = tf.math.divide(features['image'] - tf.reduce_max(features['image'], [0, 1, 2]),
                                       tf.reduce_max(features['image'], [0, 1, 2]) - tf.reduce_min(features['image'],
                                                                                                   [0, 1, 2]))
            summary.image('1_Medical_Image', image, max_outputs=1)
            summary.image('2_Output', output_img, max_outputs=1)
            summary.image('3_Output_pred', tf.expand_dims(y_pred[:, :, :, 1], -1), max_outputs=1)
            summary.image('4_Output_label', label, max_outputs=1)

    if mode == estimator.ModeKeys.TRAIN:
        with tf.name_scope('Learning_Rate'):
            global_step = tf.compat.v1.train.get_or_create_global_step()
            learning_rate = tf.compat.v1.train.exponential_decay(params['lr'], global_step=global_step,
                                                                 decay_steps=params['decay_steps'],
                                                                 decay_rate=params['decay_rate'], staircase=False)
        with tf.name_scope('Optimizer_conf'):
            train_op = Adam(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)

        with tf.name_scope('Metrics'):
            summary.scalar('Output_DSC', dice[1])
            summary.scalar('Learning_Rate', learning_rate)

    if mode == estimator.ModeKeys.EVAL:
        eval_metric_ops = {'Metrics/Output_DSC': dice}
        eval_summary_hook = tf.estimator.SummarySaverHook(output_dir=params['eval_path'],
                                                          summary_op=summary.merge_all(),
                                                          save_steps=params['eval_steps'])
        evaluation_hooks = [eval_summary_hook]

    if mode == estimator.ModeKeys.PREDICT:
        predictions_dict = {'image': features['image'],
                            'y_preds': y_pred[:, :, :, 1],
                            'output_img': output_img,
                            'path': features['path']}

    return estimator.EstimatorSpec(mode,
                                   predictions=predictions_dict,
                                   loss=loss,
                                   train_op=train_op,
                                   eval_metric_ops=eval_metric_ops,
                                   training_hooks=training_hooks,
                                   evaluation_hooks=evaluation_hooks)
