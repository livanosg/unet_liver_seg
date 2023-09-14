from os import makedirs
from tensorflow.version import VERSION
from config import ROOT_DIR


def save_logs(args, log_data):
    """ Log configuration and information of model"""
    logs = [120 * '#',
            'TensorFlow Version: {}'.format(VERSION),
            'Mode: {}'.format(args.mode),
            120 * '#',
            'Working Directory: {}'.format(ROOT_DIR),
            'Model Options',
            120 * '#',
            'You have chosen {} data'.format(args.modality),
            'You have chosen {} classes'.format(args.classes),
            120 * '#',
            'Augmentation probability: {}%'.format(args.augm_prob * 100),
            'Batch size: {}'.format(args.batch_size),
            120 * '#',
            'Training options',
            120 * '#',
            'Training epochs: {}'.format(args.epochs),
            'Train set contains {} images'.format(log_data['train_size']),
            'Steps per epoch = {} steps'.format(log_data['steps_per_epoch']),
            'Total training steps = {} steps'.format(log_data['max_training_steps']),
            'Learning Rate: {}'.format(args.lr),
            'Number of learning rate decays in a training session: {}'.format(args.decays_per_train),
            'Decay rate of learning rate: {}'.format(args.decay_rate),
            'Early stopping after {} epochs with no metric increase.'.format(args.early_stop),
            'Dropout rate: {}'.format(args.dropout),
            120 * '#',
            'Evaluation options',
            120 * '#',
            'Evaluation set contains {} examples'.format(log_data['eval_size']),
            'Total evaluation steps = {}.'.format(log_data['eval_steps']),
            120 * '#',
            'Estimator configuration',
            120 * '#',
            'Random seed: {}'.format(str(args.seed)),
            120 * '#']
    makedirs(log_data['model_path'], exist_ok=True)
    file = open(log_data['model_path'] + '/train_info.txt', "w+")
    for i in logs:
        file.write(i)
        file.write('\n')
    file.close()
