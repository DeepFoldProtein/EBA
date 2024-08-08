import torch
from eba import methods 
from eba import score_matrices as sm
from eba import plm_extractor as plm

### load language model extractor: ProtT5 or ESMb1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
protT5_ext = plm.load_extractor('ProtT5', 'residue', device=device)

### sequences example 
seq1 = 'MLIAFEGIDGSGKTTQAKKLYEYLKQKGYFVSLYREPGGTKVGEVLREILLTEELDERTELLLFEASRSKLIEEKIIPDLKRDKVVILDRFVLSTIAYQGYGKGLDVEFIKNLNEFATRGVKPDITLLLDIPVDIALRRLKEKNRFENKEFLEKVRKGFLELAKEEENVVVIDASGEEEEVFKEILRALSGVLRV'
seq2 = 'RRGALIVLEGVDRAGKSTQSRKLVEALCAAGHRAELLRFPERSTEIGKLLSSYLQKKSDVEDHSVHLLFSANRWEQVPLIKEKLSQGVTLVVDRYAFSGVAFTGAKENFSLDWCKQPDVGLPKPDLVLFLQLQLADAAKRGAFGHERYENGAFQERALRCFHQLMKDTTLNWKMVDASKSIEAVHEDIRVLSEDAIATATEKPLGELWK'

### extract per-residue embeddings (if you are using ProstT5, add "<AA2fold> " to the sequences)
emb1 = protT5_ext.extract(seq1)
emb2 = protT5_ext.extract(seq2)
print(emb1.shape)

### compute similarity matrix and EBA score
similarity_matrix = sm.compute_similarity_matrix(emb1, emb2)
eba_results = methods.compute_eba(similarity_matrix)
### to return the alignment itself use:
#eba_results = methods.compute_eba(similarity_matrix, extensive_output=True)

### show results
print('EBA raw: ', eba_results['EBA_raw'])
print('EBA min: ', eba_results['EBA_min'])
print('EBA max: ', eba_results['EBA_max'])
