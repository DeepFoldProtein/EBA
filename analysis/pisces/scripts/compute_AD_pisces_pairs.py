import sys
import os
import torch
import pickle

### inputs
model = sys.argv[1]  #ProtT5, ESMb1
p = int(sys.argv[2])

### defining paths
working_dir = '../'
results_path = os.path.join(working_dir, 'results')
embeddings_dir = os.path.join(working_dir, 'data/embeddings')
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
average_embeddings = torch.load(average_embeddings_path, map_location=torch.device('cpu'))
print('{}: {} embedding loaded'.format(model, len(average_embeddings)))

### make sure to have embeddings for each sequence
pairs_list = [p for p in pairs_list if p[0] in average_embeddings.keys() and p[1] in average_embeddings.keys()]
print('number of pairs: {}'.format(len(pairs_list)))

### compute AD
distance_metric = torch.nn.PairwiseDistance(p=2)
AD_pairs = dict()
for pair in pairs_list:
    AD_pairs[pair] = distance_metric(torch.unsqueeze(average_embeddings[pair[0]],0), 
                                        torch.unsqueeze(average_embeddings[pair[1]],0))

### save
pairs_ss_file = os.path.join(results_path, '{}_AD.pickle'.format(model))
with open(pairs_ss_file, 'wb') as handle:
    pickle.dump(AD_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
