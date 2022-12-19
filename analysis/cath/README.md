# CATH label transfer analysis

This folder contains the scripts and the data necessary to reproduce the results described in the paragraph 'EBA successfully transfers CATH annotations' of: [https://doi.org/10.1101/2022.12.13.520313](https://doi.org/10.1101/2022.12.13.520313).


### Compute embeddings
The first step of the analysis consists in computing the per-residue embeddings using the desired language model (ProtT5 or ESMb1). The necessary script can be found in [/scripts](https://git.scicore.unibas.ch/schwede/EBA/-/tree/main/analysis/cath/scripts). The embeddings will be stored in [/data/embeddings](https://git.scicore.unibas.ch/schwede/EBA/-/tree/main/analysis/cath/data/embeddings). The computational times of this process strongly benefits from the usage of a GPU.

```
python save_embeddings.py ProtT5 test219.fasta 0 219
python save_embeddings.py ProtT5 lookup69k.fasta 0 24000
python save_embeddings.py ProtT5 lookup69k.fasta 24000 48000
python save_embeddings.py ProtT5 lookup69k.fasta 48000 69605

```

### Compute EBA/AD scores
It is then possible to query a sequence of the test set against the ones in the lookup set. The necessary scripts can be found in [/scripts](https://git.scicore.unibas.ch/schwede/EBA/-/tree/main/analysis/cath/scripts). The resulting scores will be stored in [/results](https://git.scicore.unibas.ch/schwede/EBA/-/tree/main/analysis/cath/results). For example:

```
python query_against_lookupset.py 1q16B01 ProtT5
```
