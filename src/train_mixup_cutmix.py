import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

from apex import amp

import torch
from torch.utils.data import DataLoader

from utils import *
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliDataset
from augmentations import get_augmentations

fold = 0
bs = 128
n_epochs = 100
IMG_SZ = 224
use_ohem = False

model_name = 'se_resnext50'

print(f'Training {model_name}, ohem_loss={use_ohem}, fold={fold}, img={IMG_SZ}, bs={bs}, epochs={n_epochs}')

PATH = '../data/'
LOG_DIR = '../logs/'
WEIGHT_DIR = '../trained_models/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

df = pd.read_csv(f'{PATH}train.csv')
image_data = np.load(f'{PATH}train.npy')
val_mask = np.load(f'{PATH}val_mask_{fold}.npy')

nunique = list(df.nunique())[1:-1]
print(nunique, df.shape)
print(list(df[~val_mask].nunique())[1:-1])

trn_aug, val_aug = get_augmentations(IMG_SZ)

train_dataset = BengaliDataset(df[~val_mask], image_data[~val_mask], trn_aug)
valid_dataset = BengaliDataset(df[val_mask], image_data[val_mask], val_aug)

del df, image_data
gc.collect()

train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=bs, shuffle=True)

model = MODEL_DISPATCHER[model_name].to(device)
criterion = Combined_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-2, total_steps=None, epochs=n_epochs, 
                                                steps_per_epoch=len(train_dataset)//bs, 
                                                pct_start=0.1, anneal_strategy='cos', 
                                                cycle_momentum=True, base_momentum=0.8, 
                                                max_momentum=0.9,  div_factor=25.0)
model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

def train(epoch, history):
    model.train()
    running_loss, running_recall = 0.0, 0.0
    
    for idx, data_dict in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
        inputs = data_dict['image']
        labels1 = data_dict['grapheme_root']
        labels2 = data_dict['vowel_diacritic']
        labels3 = data_dict['consonant_diacritic']
        
        inputs = inputs.to(device)
        labels1, labels2, labels3 = labels1.to(device), labels2.to(device), labels3.to(device)
        
        optimizer.zero_grad()

        if np.random.rand() < 0.5:
            inputs, targets = mixup(inputs.unsqueeze(1).float(), labels1, labels2, labels3, 1.0)
            out1,out2,out3 = model(inputs)
            loss = mixup_criterion(out1,out2,out3, targets, use_ohem=use_ohem)
        else:
            inputs, targets = cutmix(inputs.unsqueeze(1).float(), labels1, labels2, labels3, 1.0)
            out1,out2,out3 = model(inputs)
            loss = cutmix_criterion(out1,out2,out3, targets, use_ohem=use_ohem)
        
        running_loss += loss.item()
        running_recall+= macro_recall_multi(out1,labels1,out2,labels2,out3,labels3)
        
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
            
        optimizer.step()
        
        scheduler.step()
        
    trn_loss = float("{:.4f}".format(running_loss/len(train_dataloader)))
    trn_recall = float("{:.4f}".format(running_recall/len(train_dataloader)))
    
    print(f'Epoch: {epoch+1}/{n_epochs}')
    print(f'trn loss: {trn_loss}, recall: {trn_recall}')
    
    torch.cuda.empty_cache()
    gc.collect()

    history.loc[epoch, 'train_loss'] = trn_loss
    history.loc[epoch,'train_recall'] = trn_recall
    
    return  trn_recall

def evaluate(epoch, history):
    model.eval()
    running_loss, running_recall = 0.0, 0.0

    with torch.no_grad():
        for idx, data_dict in enumerate(valid_dataloader):
            inputs = data_dict['image']
            labels1 = data_dict['grapheme_root']
            labels2 = data_dict['vowel_diacritic']
            labels3 = data_dict['consonant_diacritic']
        
            inputs = inputs.to(device)
            labels1, labels2, labels3 = labels1.to(device), labels2.to(device), labels3.to(device)

            out1,out2,out3 = model(inputs.unsqueeze(1).float())

            loss = criterion(out1, out2, out3, labels1, labels2, labels3)
            running_loss += loss.item()


            running_recall+= macro_recall_multi(out1,labels1,out2,labels2,out3,labels3)


    val_loss = float("{:.4f}".format(running_loss/len(valid_dataloader)))
    val_recall = float("{:.4f}".format(running_recall/len(valid_dataloader)))

    print(f'val loss: {val_loss}, recall: {val_recall}')

    history.loc[epoch, 'valid_loss'] = val_loss
    history.loc[epoch, 'valid_recall'] = val_recall
    return  val_recall

history = pd.DataFrame()
valid_recall = 0.0
best_valid_recall = 0.0

for epoch in range(n_epochs):
    torch.cuda.empty_cache()
    gc.collect()
    train_recall = train(epoch,history)
    valid_recall = evaluate(epoch,history)
    
    history.to_csv(f'{LOG_DIR}{model_name}_mixup_cutmix_{IMG_SZ}_{n_epochs}epochs_fold_{fold}.csv', index=False)
    
    if valid_recall > best_valid_recall:
        print(f'Validation recall has increased from:  {best_valid_recall:.4f} to: {valid_recall:.4f}. Saving weights.')
        torch.save(model.state_dict(), f'{WEIGHT_DIR}{model_name}_mixup_cutmix_{IMG_SZ}_{n_epochs}epochs_fold_{fold}.pth')
        best_valid_recall = valid_recall 
