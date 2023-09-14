import os
from os.path import dirname, abspath

ROOT_DIR = dirname(abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
ZIPS_DIR = os.path.join(DATA_DIR, "zips")
TRAIN_DIR = os.path.join(DATA_DIR, "Train_Sets")
EVAL_DIR = os.path.join(DATA_DIR, "Eval_Sets")
TEST_DIR = os.path.join(DATA_DIR, "Test_Sets")
PRED_DIR = os.path.join(DATA_DIR, "pred")
SAVE_DIR = os.path.join(ROOT_DIR, "saves")

CHAOS_TEST_ZIP = os.path.join(ZIPS_DIR, "CHAOS_Test_Sets.zip")
CHAOS_TRAIN_ZIP = os.path.join(ZIPS_DIR, "CHAOS_Train_Sets.zip")
