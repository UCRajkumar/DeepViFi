# DeepViFi: Detecting Viral Infections in NGS data

DeepViFi pipeline detects viral infections in NGS data using the Transformer architecture. It can be trained on specific viral families. 

Rajkumar, U. et al. *DeepViFi: Detecting Novel Integrations in NGS data.* In Review. (2022)

## Directory structure

| File             | Description                                 |
| ---------------- | ------------------------------------------- |
| config.yaml      | File to set parameters for different tasks  |
| Makefile         | Makefile to run the different scripts       |
| requirements.txt | Requirement file for setting up environment |
| setup.sh         | Script to install ecSeg package             |

...

| Folder | Description                        |
| ------ | ---------------------------------- |
| src    | Contains python scripts            |
| models | Contains transformer models trained on HPV and HBV references. Download the models.zip from here and and unzip the folder inside the deepvifi/ folder|

## Installation

```
git clone https://github.com/ucrajkumar/DeepViFi
cd DeepViFi
conda env create -f environment.yml
conda activate deepvifi
```

## Training data specifications

The training data must be a single fasta file containing all known reference genoems of the virus of interest. It must be in the form:
```
> name_of_ref
CACACATACAGACATA...
> name_of_ref2
...

The genomes do not need to be the same dimensions.
```

## Tasks
### `make create_data`
Reads the input fasta file containing the reference genomes and creates the training data to train the transformer. Set parameters in config.yaml under `create_data`:

```
datapath: Path to folder containing the viral references
viral_refs: Name of fasta file
read_len : Length of read
coverage : Simulate reads at coverage. (recommended value 0.5)
mask_rate : What portion of each input read to mask. (recommended value 0.2)
```

### `make train_transformer`
Trains the transformer given the hyperparameters. Set parameters in config.yaml under `train_transformer`:

```
datapath: Path to folder containing the viral references and training data
model_path : Path to folder where you want the model saved
embed_dim : Dimensions to embed the input read. (recommended value 128 or 256)
num_heads : Number of attention heads. (recommended value 8 or 16)
num_layers : Number of transformer blocks. (recommended value 8)
ff_dim : Dimensions of feed forward layer in each transformer block. (recommended value 256)
batch_size : Batch size
```


