from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt

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
