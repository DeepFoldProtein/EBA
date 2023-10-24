import sys
sys.path.append('../../../modules')
import similarity_matrix as sm
import EBA as eba
import os
import torch
import pickle

### function to merge stored embedding representations
def Merge(dict_1, dict_2, dict_3):
    dict_1.update(dict_2)
    dict_1.update(dict_3)
	
    return dict_1

### input
ref_sequence_id = sys.argv[1]
model = sys.argv[2]

### alignemnt parameters
gap_open_penality = 0
gap_extend_penalty = 0
l = 1.0
p = 2

### path definition
working_dir = '..'
results_path = os.path.join(working_dir, 'results/EBA_{}'.format(model))
test_embeddings_path = os.path.join(working_dir, 'data/embeddings/0_219_{}_test219_residue_embeddings.pt'.format(model))
test_average_embeddings_path = os.path.join(working_dir, 'data/embeddings/0_219_{}_test219_average_embeddings.pt'.format(model))
lookup1_embedding_path = os.path.join(working_dir, 'data/embeddings/0_24000_{}_lookup69k_residue_embeddings.pt'.format(model))
lookup2_embedding_path = os.path.join(working_dir, 'data/embeddings/24000_48000_{}_lookup69k_residue_embeddings.pt'.format(model))
lookup3_embedding_path = os.path.join(working_dir, 'data/embeddings/48000_69605_{}_lookup69k_residue_embeddings.pt'.format(model))

### load test sequence embedding
test_embedding = torch.load(test_embeddings_path, map_location=torch.device('cpu'))[ref_sequence_id]
### load lookup sequences embeddings
lookup_embeddings = Merge(torch.load(lookup1_embedding_path, map_location=torch.device('cpu')), torch.load(lookup2_embedding_path, map_location=torch.device('cpu')), torch.load(lookup3_embedding_path, map_location=torch.device('cpu')) )

### assign the preferred similarity matrix method
compute_similarity_matrix = sm.compute_similarity_matrix

### compute EBA for the test_sequence against the lookup set
EBA_pairs = dict()
for seq_id in lookup_embeddings:
    pair_ids = (ref_sequence_id, seq_id)
    similarity_matrix=compute_similarity_matrix(test_embedding,
                                                lookup_embeddings[seq_id],l=l,p=p)
    EBA_pairs[pair_ids]=eba.EBA(similarity_matrix,extensive_output=False, 
                            gap_open_penalty=gap_open_penality, 
                            gap_extend_penalty=gap_extend_penalty)


### save
isExist = os.path.exists(results_path)
if not isExist:
   os.makedirs(results_path)

pairs_ss_file = os.path.join(results_path, '{}.pickle'.format(ref_sequence_id))
with open(pairs_ss_file, 'wb') as handle:
    pickle.dump(EBA_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)