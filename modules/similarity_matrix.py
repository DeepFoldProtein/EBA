import torch

def compute_similarity_matrix(embedding1, embedding2, l=1, p=2):
    """ Take as input 2 sequence embeddings (at a residue level) and returns the similarity matrix
        with the signal enhancement based on Z-scores.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2
        :param l: scalar that can be use to regularize the similarity matrix (no effect with l=1)
        :param p: Minkowski distance order (ex. p=1:Manhattan, p=2:Euclidean)

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
        :type l: float
        :type p: integer
    """
    
    sm = torch.exp(-l*torch.cdist(embedding1, embedding2, p=p))
    columns_avg = torch.sum(sm,0)/sm.shape[0]
    rows_avg = torch.sum(sm,1)/sm.shape[1]
    
    columns_std = torch.std(sm,0)
    rows_std = torch.std(sm,1)

    z_rows = (sm-rows_avg.unsqueeze(1))/rows_std.unsqueeze(1)
    z_columns = (sm-columns_avg)/columns_std
    
    return (z_rows+z_columns)/2


def compute_similarity_matrix_plain(embedding1, embedding2, l=1, p=2):
    """ Take as input 2 sequence embeddings (at a residue level) and returns the plain
        similarity matrix.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2
        :param l: scalar that can be use to regularize the similarity matrix (no effect wit l=1)
        :param p: Minkowski distance order (ex. p=1:Manhattan, p=2:Euclidean)

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
        :type l: float
        :type p: integer
    """

    return torch.exp(-l*torch.cdist(embedding1, embedding2, p=p))

