import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from tm_vec.tm_vec_utils import featurize_prottrans, embed_tm_vec, cosine_similarity_tm
import pickle
from transformers import T5EncoderModel, T5Tokenizer
import gc

torch.cuda.is_available()

#Load the ProtTrans model and ProtTrans tokenizer
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
gc.collect()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()
print('ProtT5 loaded.')

#TM-Vec model paths
#tm_vec_swiss_model_params.ckpt
#tm_vec_swiss_model_params.json
#tm_vec_swiss_model_large.ckpt
#tm_vec_swiss_model_large_params.json
#tm_vec_cath_model_large.ckpt
#tm_vec_cath_model_large_param.json
#tm_vec_cath_model.ckpt
#tm_vec_cath_model_param.json

tm_vec_model_cpnt = "./tm_vec_swiss_model.ckpt"
tm_vec_model_config = "./tm_vec_swiss_model_params.json"

#Load the TM-Vec model
tm_vec_model_config = trans_basic_block_Config.from_json(tm_vec_model_config)
model_deep = trans_basic_block.load_from_checkpoint(tm_vec_model_cpnt, config=tm_vec_model_config)
model_deep = model_deep.to(device)
model_deep = model_deep.eval()
print('TM-vec loaded.')

### Get sequence pairs
working_dir = '../'
fasta_file = os.path.join(working_dir, 'data/sequences.fasta')
pairs_file = os.path.join(working_dir,'data/pairs.pc')

results_dir = os.path.join(working_dir, 'results/TM_vec')
results_path = os.path.join(results_dir, 'TM_vec.pickle')

### load pairs to score
pairs_list = list()
with open(pairs_file, 'r') as file:
    skip_head = True
    for line in file:
        if skip_head:
            skip_head=False
            continue
        infos = line.split()
        pairs_list.append((infos[0], infos[1]))

print('loaded {} pairs'.format(len(pairs_list)))
### parse pisces fasta and store sequences
sequences = dict()
with open(fasta_file) as file:
    for row in file:
        if row[0]=='>':
            seq_id = row.split()[0][1:]
            sequences[seq_id] = ''
        else:
            sequences[seq_id]+=row.strip()


predictions = dict()
for p in pairs_list:
    try:
        input_sequence_1 = sequences[p[0]]
        input_sequence_2 = sequences[p[1]]

        sequence_1 = np.expand_dims(input_sequence_1, axis=0)
        sequence_2 = np.expand_dims(input_sequence_2, axis=0)

        #Featurize sequence 1 and 2 using ProtTrans
        protrans_sequence_1 = featurize_prottrans(sequence_1, model, tokenizer, device).detach()
        protrans_sequence_2 = featurize_prottrans(sequence_2, model, tokenizer, device).detach()

        #Embed sequence 1 and 2 using TM-Vec, applied to the ProtTrans features
        embedded_sequence_1 = embed_tm_vec(protrans_sequence_1, model_deep, device)
        embedded_sequence_2 = embed_tm_vec(protrans_sequence_2, model_deep, device)

        #Predict the TM-score for sequence 1 and 2, using the TM-Vec embeddings
        predicted_tm_score = cosine_similarity_tm(torch.tensor(embedded_sequence_1), torch.tensor(embedded_sequence_2))
        predictions[p] = predicted_tm_score.numpy()[0]
    
    except:
        print('failed in {}'.format(p))


with open(results_path, 'wb') as handle:
    pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)