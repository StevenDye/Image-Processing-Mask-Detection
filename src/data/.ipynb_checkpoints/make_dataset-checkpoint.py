"""This module is for making the dataset"""

sys.path.insert(1, '../src/features/')
import build_features as bf

ROOT_PATH = '../data/raw/srinivasan/'
BATCH_SIZE = 50
IMG_SIZE = 50
CATEGORIES = ['with_mask', 'without_mask']

# Import and prepare data
data = bf.create_data(IMG_SIZE, CATEGORIES, ROOT_PATH)

# Create flipped augmented data

# Create rotated augmented data





# Shuffle data 
np.random.shuffle(data)



