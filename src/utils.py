import warnings
warnings.simplefilter(action='ignore')
import numpy as np
import os, random, re, sys, glob
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

tokens = "ACGTNM"
mapping = dict(zip(tokens, range(1, len(tokens)+1)))
def clean_genomes(datapath, viral_refs):
    viral_genomes = open(os.path.join(datapath, viral_refs)).readlines()
    viral_genomes = np.array([re.sub('[^ATCGN]+' ,'', i.replace('\n', '').upper()) for i in viral_genomes[1::2]])
    np.save(os.path.join(datapath, 'viral.npy'), viral_genomes)
    return 0

def compute_rc(datapath):
    viral_train = np.load(os.path.join(datapath, 'viral.npy'))
    viral_train_rc = [] #reverse compliment
    for i in viral_train:
        viral_train_rc.append(str(Seq(i).reverse_complement()))
    viral_train_rc = np.array(viral_train_rc)
    np.save(os.path.join(datapath, 'viral_rc.npy'), viral_train_rc)
    np.save(os.path.join(datapath, 'viral_train.npy'), np.concatenate((viral_train, viral_train_rc), 0))
    return 0

def gen_reads(datapath, read_len, coverage):
    genomes = np.load(os.path.join(datapath, 'viral_train.npy'))
    print('Total # of genomes: ', len(genomes))
    print('Length of largest genome: ', len(max(genomes)))
    print('Length of shortest genomes: ', len(min(genomes)))

    X = []
    coverage = 0.5
    num_reads = int((len(max(genomes, key=len)) - read_len)*coverage)
    print("Print # of reads: ", num_reads)

    for idx, i in enumerate(genomes):
        read_locs = np.random.randint(0,len(i)-read_len-1, num_reads)
        for j in read_locs:
            X.append(list(i[j : j+read_len]))

    return np.array(X)

def gen_masked_reads(X, read_len, mask_rate):
    X_masked = X.copy()
    mask_rate = 0.2
    for i in X_masked:
        for j in np.random.randint(1,read_len-2, int(read_len*mask_rate)):
            mask_trick = np.random.randint(0, 10, 1)
            if(mask_trick < 5):
                i[j] = 'M'
            elif((mask_trick >= 5) & (mask_trick <8)):
                i[j] = random.choice(['A', 'T', 'C', 'G'])
            else:
                continue
    return np.array(X_masked)

def seqs2cat(seqs, mapping=mapping):
    def categorical_encode(seq):
        seq = [mapping[i] for i in seq]
        return np.array(seq).astype('uint8')
    vecs = []
    for i in seqs:
        vecs.append(np.array(categorical_encode(i)))
    return np.array(vecs)


# Read and extract dataframe of fasta file
def read_fasta(filename, file_type='fasta'):
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq
    
    df = pd.DataFrame()
    reads = []
    ids = []
    with open(filename) as f:
        seqs = SeqIO.parse(f, file_type)
        for seq in seqs:
            reads.append(str(seq.seq))
            ids.append(seq.id)
    df['ID'] = ids
    df['seqs'] = reads
    return df

def embed_reads(transformer, data, path):
    def save_preds(predictions, counter):
        new_path = path + '_predictions' + str(counter) + '.npy'
        predictions = np.concatenate(predictions)
        print("Saving predictions...")
        np.save(new_path, predictions)
        return 0
        
    batch_size = 50
    predictions = []
    counter = 0
    print('Transformer predicting reads...')
    if(len(data)< batch_size):
        seqs = seqs2cat(data, mapping)
        predictions.append(np.mean(transformer.predict(data), axis=1)) #fix
        save_preds(predictions, counter)
        return 0

    for i in range(int(len(data)/batch_size)):
        seqs = seqs2cat(data[i*batch_size: i*batch_size+batch_size], mapping)
        predictions.append(np.mean(transformer.predict(seqs), axis=1))
        if( (i > 0) & ((i % 4000)== 0)):
            print(i)
            save_preds(predictions, counter)
            counter += 1
            del predictions
            predictions = []

    i += 1
    seqs = seqs2cat(data[i*batch_size:], mapping)
    predictions.append(np.mean(transformer.predict(seqs), axis=1))
    save_preds(predictions, counter)
    return 0