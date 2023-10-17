import os, warnings, yaml
import numpy as np
from utils import *

import warnings
warnings.filterwarnings('ignore')

def main(argv):
    config = open("config.yaml")
    var = yaml.load(config, Loader=yaml.FullLoader)['create_data']
    datapath = var['datapath']
    viral_refs = var['viral_refs']
    read_len = var['read_len']
    coverage = var['coverage']
    mask_rate = var['mask_rate']

    if(os.path.isdir(os.path.join(datapath)) == False):
        print("Input folder does not exist. Exiting...")
        sys.exit(2)

    print('Creating training data!')
    ##########
    ##########
    print('Reading all viral references...')
    clean_genomes(datapath, viral_refs)

    print('Computing reverse complement of all references...')
    compute_rc(datapath)

    print('Generating training reads...')

    X = gen_reads(datapath, read_len, coverage)
    X_masked = gen_masked_reads(X, read_len, mask_rate)

    np.save(os.path.join(datapath, '1mer.npy'), X)
    np.save(os.path.join(datapath, '1mer_masked.npy'), X_masked)

    print("Tokenized reads...")
    tokens = "ACGTNM"
    mapping = dict(zip(tokens, range(1,len(tokens)+1)))
    print(mapping)

    X_tokenized = seqs2cat(X, mapping)
    X_masked_tokenized = seqs2cat(X_masked, mapping)

    np.save(os.path.join(datapath, 'X_tokenized.npy'), X_tokenized)
    np.save(os.path.join(datapath, 'X_masked_tokenized.npy'), X_masked_tokenized)

    #sanity check
    #print(X_tokenized[:3], X_masked_tokenized[:3])
    
if __name__ == "__main__":
   main(sys.argv[1:])
