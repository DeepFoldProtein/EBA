# Embedding-based alignment (EBA)
This repository contains the implementation of the EBA method as described in: ...
 
Notice that the embedding extraction is independent from the EBA method, and any pLM can be used. However, to facilitate the application we provide a module (embedding.py) that allows the extraction of the per-residue embedding representations for the following pLMs: ProtT5 and ESM-b1.
 
Note: In case of high dimensionality embeddings (such as ESM2), we suggest to run the EBA with the parameter l=0.1 or l=0.01 to avoid precision errors.



## Getting started
Install the following modules:
```
pip install numba
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install fair-esm
pip install transformers
pip install SentencePiece
```
If you want to use a GPU, you can follow the instructions at: https://pytorch.org/ in order to install the compatible pytorch version.

## Example

To run EBA on your own sequences you can use the following code:
``` python
import sys
sys.path.append('./modules')
import torch
import EBA as eba
import similarity_matrix as sm
import embedding as emb

### load language model extractor: ProtT5 or ESMb1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
protT5_ext = emb.load_extractor('ProtT5', 'residue', device=device)

### sequences example
seq1 = 'MLIAFEGIDGSGKTTQAKKLYEYLKQKGYFVSLYREPGGTKVGEVLREILLTEELDERTELLLFEASRSKLIEEKIIPDLKRDKVVILDRFVLSTIAYQGYGKGLDVEFIKNLNEFATRGVKPDITLLLDIPVDIALRRLKEKNRFENKEFLEKVRKGFLELAKEEENVVVIDASGEEEEVFKEILRALSGVLRV'
seq2 = 'RRGALIVLEGVDRAGKSTQSRKLVEALCAAGHRAELLRFPERSTEIGKLLSSYLQKKSDVEDHSVHLLFSANRWEQVPLIKEKLSQGVTLVVDRYAFSGVAFTGAKENFSLDWCKQPDVGLPKPDLVLFLQLQLADAAKRGAFGHERYENGAFQERALRCFHQLMKDTTLNWKMVDASKSIEAVHEDIRVLSEDAIATATEKPLGELWK'

### extract per-residue embeddings
emb1 = protT5_ext.extract(seq1)
emb2 = protT5_ext.extract(seq2)

### compute similarity matrix and EBA score
similarity_matrix = sm.compute_similarity_matrix(emb1, emb2)
eba_results = eba.EBA(similarity_matrix)
### to return the alignment itself use:
#eba_results = eba.EBA(similarity_matrix, extensive_output=True)

### show results
print('EBA raw: ', eba_results['EBA_raw'])
print('EBA min: ', eba_results['EBA_min'])
print('EBA max: ', eba_results['EBA_max'])
```
