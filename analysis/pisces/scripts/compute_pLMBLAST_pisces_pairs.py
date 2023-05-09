#import sys
#sys.path.append('/scicore/home/schwede/pantol0000/repositories/pLM-BLAST')
import os
import torch
import pickle
#import numpy as np
import torch
import alntools as aln

### inputs
model = 'ProtT5' #sys.argv[1], ESMb1, ESM2

### defining paths
working_dir = '../'
results_path = os.path.join(working_dir, 'results/pLM_BLAST')
embeddings_dir = os.path.join(working_dir, 'data/embeddings')
residue_embeddings_path = os.path.join(embeddings_dir, '{}_pisces_residue_embeddings.pt'.format(model))
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

EBA_pairs = dict()
for pair in pairs_list:
    # calculate embedding similarity aka substitution matrix
    densitymap = aln.density.embedding_similarity(residues_embeddings[pair[0]],residues_embeddings[pair[1]])
    # convert to numpy array
    densitymap = densitymap.cpu().numpy()
    # find all alignment possible paths (traceback from borders)
    paths = aln.alignment.gather_all_paths(densitymap)
    # score those paths
    results = aln.prepare.search_paths(densitymap, paths=paths, as_df=True)
    # remove redundant hits
    results = aln.postprocess.filter_result_dataframe(results)
    #get the score for the best alignment
    EBA_pairs[pair] = results.iloc[0]['score']


pairs_ss_file = os.path.join(results_path, 'pLM_BLAST.pickle')
with open(pairs_ss_file, 'wb') as handle:
    pickle.dump(EBA_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)