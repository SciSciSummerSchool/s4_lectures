# import the necessary packages
import os



# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "/home/jovyan/datasets/s4/ImageClassification/"

# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "chart_classification_validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# define the amount of data that will be used training
#TRAIN_SPLIT = 0.8

# the amount of validation data will be a percentage of the
# *training* data
#VAL_SPLIT = 0.2

# define the names of the classes
CLASSES = ['Area','Bar','Box','Heatmap','Line','Scatter','Violin']

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
#BS = 32
BS = 32
#NUM_EPOCHS = 20
NUM_EPOCHS = 1

# define the path to the serialized output model after training
MODEL_PATH = "chart_classfication_basic.model"