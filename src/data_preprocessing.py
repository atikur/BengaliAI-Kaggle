import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

PATH = '../data/'
df = pd.read_csv(f'{PATH}train.csv')

IMG_HEIGHT = 137
IMG_WIDTH = 236

n_file = 4
n_image = df.shape[0]
train_values = np.zeros((n_image, IMG_HEIGHT * IMG_WIDTH), dtype = 'uint8')

for i in tqdm(range(n_file)):
    directory = f'{PATH}train_image_data_{i}.parquet'
    train_f = pd.read_parquet(directory, engine = 'pyarrow')
    train_f.set_index('image_id', inplace=True)
    n_file_image = train_f.shape[0]
    train_values[i * n_file_image : (i + 1) * n_file_image, :] = (255 - train_f.values)
    del train_f
    gc.collect()

train_values = train_values.reshape((-1, IMG_HEIGHT, IMG_WIDTH))
np.save(f'{PATH}train.npy', train_values)