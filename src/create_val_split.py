import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

PATH = '../data/'
df = pd.read_csv(f'{PATH}train.csv')

nfold = 12

df['id'] = df['image_id'].apply(lambda x: int(x.split('_')[1]))
X, y = df[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values[:,0], df.values[:,1:]
df['fold'] = np.nan

mskf = MultilabelStratifiedKFold(n_splits=nfold, random_state=12)
for i, (_, test_index) in enumerate(mskf.split(X, y)):
    df.iloc[test_index, -1] = i
    
df['fold'] = df['fold'].astype('int')

# test set contains unseen graphemes, so let's make sure each fold contains
# few unseen graphemes
val_graphemes = {
    'val_grapheme_0': ['ক্যা','মী','চ্যূ','র্খে','টা','গ্রে','র্বি','তৈ','জ্যা','র্ভে',],
    'val_grapheme_1': ['কৃ','র্মা','র্চে','খ্রী','ট্র্যা','গে','ব্যূ','তৌ','জি','ভ্যে',],
    'val_grapheme_2': ['ক্য','মৈ','চা','র্খ','ট্যু','গু','বে','তৃ','র্জি','র্ভা',],
    'val_grapheme_3': ['কৈ','র্ম','চ্যা','খী','ট্রে','গ্যা','বা','তেঁ','জ্য','ভৃ',],
    'val_grapheme_4': ['কো','ম্যা','চী','খে','র্টা','গৈ','বি','র্ত','জ্যে','ভূ',],
    'val_grapheme_5': ['র্ক','মূ','চো','খু','ট্রো','গৃ','বৈ','র্তে','জ্যি','র্ভি',],
    'val_grapheme_6': ['র্কো','মি','চেঁ','খো','টূ','গা','র্ব্য','ত্যা','জ্যৈ','ভৈ',],
    'val_grapheme_7': ['কী','মো','চ্য','র্খা','ট্যে','গি','ব্যা','র্তা','জা','ভৌ',],
    'val_grapheme_8': ['কে','মা','র্চ','খ্যা','টী','গ্রা','র্বো','ত্যি','র্জ্য','র্ভু',],
    'val_grapheme_9': ['কি','ম্যে','চু','খা','র্ট','র্গা','বী','তি','জ্র','ভ্য',],
    'val_grapheme_10': ['ক্যু','মে','র্চি','খি','র্টে','গ্রী','ব্রা','ত্যে','র্জু','ভো',],
    'val_grapheme_11': ['কেঁ','মু','র্চা','খ্য','টি','গ্যে','ব্রী','র্তৃ','র্জী','ভ্যা',],
}

def save_val_mask(fold):
    val_idx1 = df[df.fold == fold].index.tolist()
    print(len(val_idx1), val_idx1[:10])

    # these samples has unseen graphemes
    val_idx2 = df[df.grapheme.isin(val_graphemes[f'val_grapheme_{fold}'])].index.tolist()
    print(len(val_idx2))

    val_idx = val_idx1 + val_idx2
    val_idx = list(set(val_idx))

    print(len(val_idx), sorted(val_idx)[-5:])

    val_mask = np.zeros(df.shape[0],dtype=bool)
    val_mask[val_idx] = True

    np.save(f'{PATH}val_mask_{fold}.npy', val_mask)

for fold in range(nfold):
    save_val_mask(fold)
    print('')

del df
gc.collect()
