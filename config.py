from os.path import dirname, abspath

root_dir = dirname(abspath(__file__))
dataset_root = root_dir + '/Datasets'

paths = {'train': dataset_root + '/Train',
         'eval': dataset_root + '/Eval',
         'chaos-test': dataset_root + '/Test_Sets',
         'predict': dataset_root + '/Predict',
         'save': root_dir + '/saves'}