import sys
sys.path.append('../../../modules')
import os
import embedding as emb
import torch

model = sys.argv[1]

working_dir = '../'
data_dir = os.path.join(working_dir, 'data')
sequence_file = os.path.join(data_dir, 'sequences.fasta')
res_embedding_path = os.path.join(data_dir, 'embeddings/{}_pisces_residue_embeddings.pt'.format(model))
avg_embedding_path = os.path.join(data_dir, 'embeddings/{}_pisces_average_embeddings.pt'.format(model))

#load extractors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
res_extractor = emb.load_extractor(model, 'residue', device=device)
avg_extractor = emb.load_extractor(model, 'avg', device=device)

#load sequences
sequences = dict()
with open(sequence_file) as file:
    for row in file:
        if row[0]=='>':
            seq_id = row.split()[0][1:]
            sequences[seq_id] = ''
        else:
            sequences[seq_id]+=row.strip()
        
#compute embeddings
res_embedding = dict()
avg_embedding = dict()
n=0
skipped = 0
for name in sequences.keys():
    n+=1
    seq = sequences[name]
    try:
        res_embedding[name] = res_extractor.extract(seq)
        avg_embedding[name] = avg_extractor.extract(seq)
    except Exception as error:
        skipped+=1
        print('Skipping {}, length: {}'.format(name, len(seq)))
        print(error)

    if n%100==0:
        print('computed {} embeddings so far...'.format(n))

print('extracted {} embeddings'.format(n-skipped))
print('skipped {} sequences'.format(skipped))

#store embeddings
torch.save(res_embedding, res_embedding_path)
torch.save(avg_embedding, avg_embedding_path)

