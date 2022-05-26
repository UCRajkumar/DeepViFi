import pickle,os, yaml, logging, warnings, sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformer import *
from utils import *

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
print("Tensorflow version: ", tf.__version__)

import warnings
warnings.filterwarnings('ignore')

def main(argv):
    config = open("config.yaml")
    var = yaml.load(config, Loader=yaml.FullLoader)['train_transformer']
    datapath = var['datapath']
    model_path = var['model_path']
    ck_name = var['ck_name']
    embed_dim = var['embed_dim']
    num_heads = var['num_heads']
    num_layers = var['num_layers']
    ff_dim = var['ff_dim']
    batch_size = var['batch_size']

    if(os.path.isdir(os.path.join(datapath)) == False):
        print("Input folder does not exist. Exiting...")
        sys.exit(2)

    plot_path = os.path.join(model_path, 'plots')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    model_ck_path = os.path.join(model_path, ck_name) #for having multiple versions
    if not os.path.exists(model_ck_path):
        os.makedirs(model_ck_path)

    strategy = tf.distribute.MirroredStrategy()

    X_tokenized = np.load(os.path.join(datapath, 'X_tokenized.npy'))
    X_masked_tokenized = np.load(os.path.join(datapath, 'X_masked_tokenized.npy'))
    print("Training and GT data shapes: ", X_tokenized.shape, ', ', X_masked_tokenized.shape)

    tokens = "ACGTNM"
    mapping = dict(zip(tokens, range(0,len(tokens)+1)))
    print(mapping)

    maxlen = X_tokenized.shape[-1]
    vocab_size = len(tokens)+1
    final_dense = vocab_size-2
    print('Max read length: ', maxlen)
    print('Vocab size: ', vocab_size)

    # if you want to save model after every epoch, uncomment the next line
    #saver = CustomSaver(model_ck_path)

    es = EarlyStopping(monitor='loss', min_delta=0.0005, verbose=2, patience=10)

    model_name = 'embed' + str(embed_dim) + '_heads' + str(num_heads) + '_h' + str(ff_dim) + '_fd' + str(final_dense) + '_nl' + str(num_layers)
    print(model_name)
    with strategy.scope():
        lr = CustomSchedule(ff_dim)
        optimizer = tf.keras.optimizers.Adam(lr) #Sometimes using just a consistent lr of 0.0001 works very well
        
        inputs = layers.Input(shape=(maxlen,))
        embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        print(maxlen, vocab_size, embed_dim)
        x = embedding_layer(inputs)
        
        for i in range(num_layers):
            transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
            x = transformer_block(x)
        
        outputs = layers.Dense(final_dense, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss= "sparse_categorical_crossentropy")
        print(model.summary())

    history = model.fit(X_masked_tokenized, X_tokenized, batch_size=batch_size, verbose = 1, epochs=100)
    
    with open(os.path.join(model_ck_path, model_name + '_history'), 'wb') as f:
        pickle.dump(history.history, f)

    model.save(os.path.join(model_ck_path, model_name), save_format='tf')

    fig, ax1 = plt.subplots()
    ax1.plot(history.history['loss'], color='red', label='loss')
    fig.savefig(os.path.join(model_ck_path, model_name + '_loss.png'))

if __name__ == "__main__":
   main(sys.argv[1:])