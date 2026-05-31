from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from surprise import NMF
from surprise import accuracy

from surprise.model_selection import train_test_split

import pandas as pd

# Load dataset
data = Dataset.load_builtin('ml-1m')

# Convert to DataFrame
df = pd.DataFrame(
    data.raw_ratings,
    columns=[
        'user_id',
        'item_id',
        'rating',
        'timestamp'
    ]
)

df = df.drop(columns=['timestamp'])

# Reader
reader = Reader(
    rating_scale=(1,5)
)

# Reload dataset
data = Dataset.load_from_df(
    df[['user_id','item_id','rating']],
    reader
)

# Split
trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

# ----------------------
# KNN MODEL
# ----------------------

sim_options = {

    'name':'pearson',

    'user_based':True

}

model_KNN = KNNBasic(
    sim_options=sim_options
)

model_KNN.fit(trainset)

predictions = model_KNN.test(
    testset
)

print("\nKNN Results")

accuracy.rmse(predictions)

# ----------------------
# NMF MODEL
# ----------------------

model_NMF = NMF()

model_NMF.fit(trainset)

predictions = model_NMF.test(
    testset
)

print("\nNMF Results")

accuracy.rmse(predictions)