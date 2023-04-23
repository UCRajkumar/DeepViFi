import pickle,os, yaml, logging, warnings, sys
from joblib import dump, load
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from utils import *


logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
print("Tensorflow version: ", tf.__version__)

import warnings
warnings.filterwarnings('ignore')

def main(argv):
    config = open("config.yaml")
    var = yaml.load(config, Loader=yaml.FullLoader)['test_pipeline']
    datapath = var['datapath']
    model_path = var['model_path']
    ck_name = var['ck_name']
    transformer_model = var['transformer_model']
    rf_model = var['rf_model']
    test_file = var['test_file']
    file_type = var['file_type']

    if(os.path.isdir(os.path.join(datapath)) == False):
        print("Input folder does not exist. Exiting...")
        sys.exit(2)

    print('Load and process test data...')
    reads = read_fasta(os.path.join(datapath, test_file), file_type)
    reads.to_csv(os.path.join(datapath, test_file.split('.')[0] +'.csv'))
    del reads

    print("Loading transformer model...")
    transformer = load_model(os.path.join(model_path, ck_name, transformer_model), compile=False)
    transformer.summary()
    transformer = Model(transformer.input, transformer.layers[-2].output)

    seqs = pd.read_csv(os.path.join(datapath, test_file.split('.')[0] +'.csv'), index_col=0).seqs
    print("Number of reads: ", len(seqs))
    embed_reads(transformer, seqs, os.path.join(datapath, test_file.split('.')[0]))
    del seqs
    
    rf = load(os.path.join(model_path, ck_name, rf_model))

    preds_1 = []
    for i in np.sort(glob.glob(os.path.join(datapath, '*predictions*.npy'))):
        preds_1.append(np.load(i))
    preds_1 = np.concatenate(preds_1)   

    rf_preds_1 = rf.predict_proba(preds_1)[:, 1]

    seqs = pd.read_csv(os.path.join(datapath, test_file.split('.')[0] +'.csv'), index_col=0)
    seqs = seqs.drop(columns=['seqs'])
    seqs['rf_preds'] = rf_preds_1
    seqs.to_csv(os.path.join(datapath, test_file.split('.')[0] +'_rfpredictions.csv'))

    new_path = os.path.join(datapath, test_file.split('.')[0] + '_rfpredictions.npy')
    np.save(new_path, rf_preds_1)

if __name__ == "__main__":
   main(sys.argv[1:])
