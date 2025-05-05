import torch
from torch.utils.data import Dataset


class PersistenceTransformDataset(Dataset):
    
    def __init__(self, dataset_base, diagrams, idx=None, eps=None, return_X=False):
        super().__init__()
        
        self.return_X = return_X
        if self.return_X:
            self.X = dataset_base.data
        self.y = dataset_base.targets
        
        # get diagrams as list of tensors
        D = torch.ones([len(diagrams), max(map(len, diagrams))+1, 4]) * torch.inf

        # select points according to eps and idx
        for i, dgm in enumerate(diagrams):

            # eps
            if eps is not None:
                eps_idx = (dgm[:,1] - dgm[:,0]) >= eps
                dgm = dgm[eps_idx]

            # idx
            if idx is not None:
                dgm_idx = torch.isin(dgm[:,-1], idx)
                dgm = dgm[dgm_idx]
                D[i,:len(dgm)] = dgm[:,:-1]
            else:
                D[i,:len(dgm)] = dgm[:,:-1]

        # cut to the largest diagram accross all dataset
        if idx is not None:
            max_len = torch.argmax(D[:,:,0], axis=1).max()
            D = D[:,:max_len+1] # leave at least one inf value!
            
        self.D = D
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx] if self.return_X else None, self.D[idx], self.y[idx]
    

def collate_fn(batch):

    # get len of a batch and len of each diagram in a batch
    n_batch = len(batch)
    d_lengths = [int(torch.argmax(D[:,0])) for X_, D, y_ in batch]
    
    # set batch tensor to the max length of a diagram in a batch
    Ds = torch.ones([n_batch, max(d_lengths), 4]) * 0.
    D_masks = torch.zeros([n_batch, max(d_lengths)]).bool()
    ys = torch.zeros(n_batch).long()
    
    # populate data, targets, diagrams and their masks
    for i, (X_, D, y) in enumerate(batch):
        Ds[i][:d_lengths[i]] = D[:d_lengths[i]]
        D_masks[i][d_lengths[i]:] = True
        ys[i] = y
    
    return Ds, D_masks, ys