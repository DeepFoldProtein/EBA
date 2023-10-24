#ml OpenStructure/2.3.0-foss-2018b
#ml HH-suite/3.2.0-foss-2018b-Boost-1.68.0-Python-3.6.6

import os
import shutil
from ost import seq
from ost.bindings import hhblits3

working_dir = '../'
data_dir = os.path.join(working_dir, 'data')
fasta_file = os.path.join(data_dir, 'data/sequences.fasta')

sequences = dict()
with open(fasta_file, 'r') as file:
    for row in file:
        if row[0]=='>':
            seq_id = row.split()[0][1:]
            sequences[seq_id] = ''

        elif seq_id in sequences.keys():
            sequences[seq_id]+=row.strip()

print('Retrived {} sequences'.format(len(sequences)))

for seq_id in sequences:
    print('Computing {} profile'.format(seq_id))
    my_s = seq.CreateSequence(seq_id, sequences[seq_id])
    nrdb = "/scicore/data/managed/Uniclust/latest/uniclust30_2018_08/uniclust30_2018_08"

    hh = hhblits3.HHblits(my_s, hhsuite_root=os.getenv('EBROOTHHMINSUITE'))
    a3m = hh.BuildQueryMSA(nrdb, options={'n': 2, 'cpu': 8})

    # the a3m file lives in a temporary directory, lets copy it somewhere save
    output_path = os.path.join(working_dir, 'data/a3m_profiles/{}.a3m'.format(seq_id))
    shutil.copy(a3m, output_path)