from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not Path.exists(tarball_path):
        Path.mkdir(Path("datasets"), parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
# Set display options to show all columns and rows nicely
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 5)
print("\nFirst 5 rows of the housing dataset:")
print("=====================================")
print(housing.head())
print("\nDataset info:")
print("=============")
print(housing.info())

print("\nOcean proximity value counts:")
print("============================")
print(housing['ocean_proximity'].value_counts())

print("\nDescription of the dataset:")
print("===========================")
pd.set_option('display.max_rows', None)
print(housing.describe())


# Set style for better readability
plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Create histograms with enhanced styling
housing.hist(bins=50, figsize=(20,15))

# Adjust layout to prevent label cutoff
plt.tight_layout(pad=1.5)

# Add a title to the entire figure
plt.suptitle('Distribution of Housing Features', fontsize=20, y=1.02)

# Save and display the plot
plt.show()

def shuffle_and_split_data(housing, test_ratio):
    np.random.seed(42)  # Set random seed for reproducibility
    shuffled_indices = np.random.permutation(len(housing))
    test_set_size = int(len(housing) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return housing.iloc[train_indices], housing.iloc[test_indices]

train_set, test_set = shuffle_and_split_data(housing, 0.2)

print("\nDataset split sizes:")
print("===================")
print(f"Training set size: {len(train_set)} samples")
print(f"Test set size:     {len(test_set)} samples")

def test_set_check(identifier, test_ratio):
    """Return True if the instance should be in the test set based on its identifier hash."""
    return hash(np.int64(identifier)) & 0xFFFFFFFF <= test_ratio * 0xFFFFFFFF

def split_data_with_id_hash(data, id_column, test_ratio):
    """Split the data into train and test sets using a hash of the id column.
    
    Args:
        data: pandas DataFrame containing the dataset
        id_column: name of the column to use as identifier
        test_ratio: proportion of data to put in test set (between 0 and 1)
    """
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
# Create a unique identifier by combining longitude and latitude
housing['id'] = housing['longitude'].astype(str) + '_' + housing['latitude'].astype(str)
train_set, test_set = split_data_with_id_hash(housing, "id", 0.2)

# Save train and test sets to CSV files
train_set.to_csv("datasets/housing/housing_train.csv", index=False)
test_set.to_csv("datasets/housing/housing_test.csv", index=False)

print("\nDataset split sizes using hash:")
print("==============================")
print(f"Training set size: {len(train_set)} samples")
print(f"Test set size:     {len(test_set)} samples")
print("\nDatasets saved as 'housing_train.csv' and 'housing_test.csv'")
