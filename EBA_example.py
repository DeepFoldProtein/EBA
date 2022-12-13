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
