import torch
from torch.utils.data import Dataset

class BengaliDataset(Dataset):

    IMG_ACTUAL_HEIGHT = 137
    IMG_ACTUAL_WIDTH = 236
    
    def __init__(self, df, image_data, aug):
        self.df, self.image_data, self.aug = df, image_data, aug
        
        self.grapheme_root = self.df.grapheme_root.values
        self.vowel_diacritic = self.df.vowel_diacritic.values
        self.consonant_diacritic = self.df.consonant_diacritic.values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img = self.image_data[idx].reshape(self.IMG_ACTUAL_HEIGHT, self.IMG_ACTUAL_WIDTH)

        if self.aug is not None:
            img = self.aug(image=img)['image']

        img = img/255.
        
        return {
            'image': torch.tensor(img, dtype=torch.float),
            'grapheme_root': torch.tensor(self.grapheme_root[idx], dtype=torch.long),
            'vowel_diacritic': torch.tensor(self.vowel_diacritic[idx], dtype=torch.long),
            'consonant_diacritic': torch.tensor(self.consonant_diacritic[idx], dtype=torch.long),
        }