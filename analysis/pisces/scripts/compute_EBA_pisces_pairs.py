import sys
sys.path.append('../../../modules')
import similarity_matrix as sm
import EBA as eba
import os
import torch
import pickle

### inputs
model = sys.argv[1]  #ProtT5, ESMb1
gap_open_penality = float(sys.argv[2])
gap_extend_penalty = float(sys.argv[3])
l = float(sys.argv[4])
p = int(sys.argv[5])

### defining paths
working_dir = '../'
results_path = os.path.join(working_dir, 'results')
embeddings_dir = os.path.join(working_dir, 'data/embeddings')
residue_embeddings_path = os.path.join(embeddings_dir, '{}_pisces_residue_embeddings.pt'.format(model))
average_embeddings_path = os.path.join(embeddings_dir, '{}_pisces_average_embeddings.pt'.format(model))
pairs_file = os.path.join(working_dir,'data/pairs.pc')

### load pairs of sequences to score
pairs_list = list()
with open(pairs_file, 'r') as file:
    skip_head = True
    for line in file:
        if skip_head:
            skip_head=False
            continue
        infos = line.split()
        if (infos[1], infos[0]) not in pairs_list:
            pairs_list.append((infos[0], infos[1]))

### load pre computed per residue embeddings
residues_embeddings = torch.load(residue_embeddings_path, map_location=torch.device('cpu'))

### make sure to have embeddings for each sequence
pairs_list = [p for p in pairs_list if p[0] in residues_embeddings.keys() and p[1] in residues_embeddings.keys()]
print('number of pairs: {}'.format(len(pairs_list)))

### assign the preferred similarity matrix (altrenative: sm.compute_similarity_matrix_plain)
compute_similarity_matrix = sm.compute_similarity_matrix

### compute EBA
EBA_pairs = dict()
n=0
for pair in pairs_list:
    similarity_matrix=compute_similarity_matrix(residues_embeddings[pair[0]],residues_embeddings[pair[1]], l=l, p=p)
    EBA_pairs[pair]=eba.EBA(similarity_matrix, extensive_output=False,gap_open_penalty=gap_open_penality, gap_extend_penalty=gap_extend_penalty)
    n+=1
    if n%100==0:
        print('computed {} EBAs so far...'.format(n))

### save
pairs_ss_file = os.path.join(results_path, '{}_EBA.pickle'.format(model))
with open(pairs_ss_file, 'wb') as handle:
    pickle.dump(EBA_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
