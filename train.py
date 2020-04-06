import os
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.logging import info
from tensorflow.estimator.experimental import stop_if_no_decrease_hook
import data_handling
from model_fn import unet_model_fn
from config import paths
from logs_script import save_logs


def train(args):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    # Distribution Strategy
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # TODO Implement on multi-nodes SLURM
    strategy = tf.distribute.MirroredStrategy()
    session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True  # Allow full memory usage of GPU.
    warm_start = None
    if args.load_model:  # Load model
        model_path = paths['save'] + '/' + args.load_model
        eval_path = model_path + '/eval'
    else:
        trial = 0
        while os.path.exists(paths['save'] + '/{}_trial_{}'.format(args.modality, trial)):
            trial += 1
        model_path = paths['save'] + '/{}_trial_{}'.format(args.modality, trial)
        eval_path = model_path + '/eval'

    train_input_fn = data_handling.DatasetHandler('train', args)
    eval_input_fn = data_handling.DatasetHandler('eval', args)

    train_size = len(train_input_fn)
    eval_size = len(eval_input_fn)
    if args.mode == 'test':
        train_size = 20
        eval_size = 10
    steps_per_epoch = np.ceil(train_size / args.batch_size)
    max_training_steps = args.epochs * steps_per_epoch

    model_fn_params = {'batch_norm': args.no_bn, 'dropout': args.dropout, 'classes': args.classes, 'lr': args.lr,
                       'decay_rate': args.decay_rate, 'decay_steps': np.ceil(args.epochs * steps_per_epoch / (args.decays_per_train + 1)),
                       'eval_path': eval_path, 'eval_steps': eval_size}

    configuration = tf.estimator.RunConfig(tf_random_seed=args.seed,
                                           save_summary_steps=steps_per_epoch,
                                           keep_checkpoint_max=args.early_stop + 2,
                                           save_checkpoints_steps=steps_per_epoch,
                                           log_step_count_steps=np.ceil(steps_per_epoch / 2),
                                           train_distribute=strategy,
                                           session_config=session_config)
    liver_seg = tf.estimator.Estimator(model_fn=unet_model_fn, model_dir=model_path, params=model_fn_params,
                                       config=configuration, warm_start_from=warm_start)

    es_steps = steps_per_epoch * args.early_stop
    early_stopping = stop_if_no_decrease_hook(liver_seg, metric_name='loss', max_steps_without_decrease=es_steps)
    profiler_hook = tf.estimator.ProfilerHook(save_steps=int(max_training_steps/5), show_memory=True, output_dir=model_path)

    log_data = {'train_size': train_size, 'steps_per_epoch': steps_per_epoch,
                'max_training_steps': max_training_steps, 'eval_size': eval_size,
                'eval_steps': eval_size, 'model_path': model_path}

    save_logs(args, log_data)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn.input_fn(),
                                        hooks=[profiler_hook, early_stopping], max_steps=max_training_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn.input_fn(),
                                      steps=eval_size, start_delay_secs=0, throttle_secs=0)
    tf.estimator.train_and_evaluate(liver_seg, train_spec=train_spec, eval_spec=eval_spec)
    info('Train and Evaluation Mode Finished!\n Metrics and checkpoints are saved at:'
         '\n {}\n ----------'.format(model_path))
