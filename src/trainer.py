import torch
import numpy as np
from tqdm import tqdm

from utils import moving_average

cuda = "cuda:0"
device = torch.device(cuda) if torch.cuda.is_available() else torch.device("cpu")

def fit(model, optimizer, loss_fn, metric_fn, n_epochs, dataloader_train, dataloader_test, desc=None):
    
    history = np.zeros((n_epochs, 4, 2)) # n_epochs, n_metrics, train/test
    
    model.to(device)
    
    pbar = tqdm(range(n_epochs), desc=desc, bar_format="{desc:<12.12}{percentage:3.0f}%|{bar:5}{r_bar}")
    for epoch_idx in pbar:
        
        # train
        model.train()
        
        loss_batches = np.zeros((len(dataloader_train), 4))
        for i, (X, mask, Y) in enumerate(dataloader_train):
            X, mask, Y = X.to(device), mask.to(device), Y.to(device)
            
            Y_hat = model(X, mask)
            loss_batch = loss_fn(Y_hat, Y)
            
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            with torch.no_grad():
                loss_batches[i,0] = loss_batch.detach() # CE
                loss_batches[i,1] = metric_fn(Y_hat, Y).detach() # accuracy
        history[epoch_idx,:,0] = loss_batches.mean(axis=0)
        
        # test
        model.eval()

        loss_batches = np.zeros((len(dataloader_test), 4))
        for i, (X, mask, Y) in enumerate(dataloader_test):
            X, mask, Y = X.to(device), mask.to(device), Y.to(device)
            Y_hat = model(X, mask)
            loss_batches[i,0] = loss_fn(Y_hat, Y).detach() # CE
            loss_batches[i,1] = metric_fn(Y_hat, Y).detach() # accuracy
        history[epoch_idx,:,1] = loss_batches.mean(axis=0)
        
        pbar.set_postfix_str("t={:.4f}, t*={:.4f}, test={:.4f}, test*={:.4f}, test@t*={:.4f}, acc={:.4f}, acc*={:.4f}, acc@t*={:.4f}".format(
            history[epoch_idx,0,0], # train
            np.min(history[:epoch_idx+1,0,0]), # train*
            history[epoch_idx,0,1], # test
            np.min(history[:epoch_idx+1,0,1]), # test*
            history[np.argmin(history[:epoch_idx+1,0,0]),0,1], # test@train*
            history[epoch_idx,1,1], # acc
            np.max(history[:epoch_idx+1,1,1]), # acc*
            history[np.argmin(history[:epoch_idx+1,0,0]),1,1], # acc@train*
            #history[np.argmin(moving_average(history[:epoch_idx+1,0,0])),0,1] # test@train**
        ))
        
    return model, history