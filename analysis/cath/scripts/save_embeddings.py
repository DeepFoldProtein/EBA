import sys
sys.path.append('../../../modules')
import os
import embedding as emb
import torch

model = sys.argv[1]
file_name = sys.argv[2]
low_idx = int(sys.argv[3])
high_idx = int(sys.argv[4])

working_dir = '..'
data_dir = os.path.join(working_dir, 'data')
sequence_file = os.path.join(data_dir, file_name)
res_embedding_path = os.path.join(data_dir, 'embeddings/{}_{}_{}_{}_residue_embeddings.pt'.format(low_idx, high_idx, model, file_name.split('.')[0]))

#load extractors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
res_extractor = emb.load_extractor(model, 'residue', device=device)

sequences = dict()
with open(sequence_file, 'r') as file:
    cath_id = ''
    for line in file:
        if line[0]=='>':
            cath_id = line[1:].strip()
            sequences[cath_id] = ''
            continue    
        sequences[cath_id] = line.strip()


cath_ids = [k for k in sequences.keys()]
cath_ids = cath_ids[low_idx:high_idx]

res_embedding = dict()
n=0
skipped = 0
for name in cath_ids:
    n+=1
    seq = sequences[name]
    try:
        res_embedding[name] = res_extractor.extract(seq)
    except Exception as error:
        skipped+=1
        print('Skipping {}'.format(name))
        print(error)


print('extracted {} embeddings'.format(n-skipped))
print('skipped {} sequences'.format(skipped))

torch.save(res_embedding, res_embedding_path)

