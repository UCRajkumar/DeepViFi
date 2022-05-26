import pickle,os, yaml, logging, warnings, sys
from joblib import dump, load
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.ensemble import RandomForestClassifier
from utils import *


logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
print("Tensorflow version: ", tf.__version__)

import warnings
warnings.filterwarnings('ignore')

def main(argv):
    config = open("config.yaml")
    var = yaml.load(config, Loader=yaml.FullLoader)['train_rf']
    datapath = var['datapath']
    model_path = var['model_path']
    ck_name = var['ck_name']
    transformer_model = var['transformer_model']
    num_trees = var['num_trees']

    if(os.path.isdir(os.path.join(datapath)) == False):
        print("Input folder does not exist. Exiting...")
        sys.exit(2)

    print("Loading transformer model")
    transformer = load_model(os.path.join(model_path, ck_name, transformer_model), compile=False)
    transformer.summary()
    transformer = Model(transformer.input, transformer.layers[-2].output)
    
    print('Loading and creating training data')
    viral_train = np.load(os.path.join(datapath, 'X_tokenized.npy'))
    idx = (list(set(np.random.randint(viral_train.shape[0], size=5000))))

    viral_train = viral_train[idx]

    viral_tr_pred = []
    batch_size = 100
    for i in range(int(viral_train.shape[0]/batch_size)):
        viral_tr_pred.append(np.mean(transformer.predict(viral_train[i*batch_size : (i+1)*batch_size]), axis=1))        
    viral_tr_pred = np.concatenate(np.array(viral_tr_pred))

    rand_train = np.random.randint(1, 5, (5000, 150))
    rand_tr_pred = []
    for i in range(int(rand_train.shape[0]/batch_size)):
        rand_tr_pred.append(np.mean(transformer.predict(rand_train[i*batch_size : (i+1)*batch_size]), axis=1))
    rand_tr_pred = np.concatenate(np.array(rand_tr_pred))

    train_pred = np.concatenate((viral_tr_pred, rand_tr_pred))
    train_gt = np.concatenate((np.ones(viral_tr_pred.shape[0]), np.zeros(rand_tr_pred.shape[0])))

    print("Train RF model...")
    rf = RandomForestClassifier(n_estimators=num_trees, warm_start=True, random_state=0)
    rf.fit(train_pred, train_gt)

    print("Save RF model...")
    dump(rf, os.path.join(model_path, ck_name, 'rf_detector.joblib')) 


if __name__ == "__main__":
   main(sys.argv[1:])