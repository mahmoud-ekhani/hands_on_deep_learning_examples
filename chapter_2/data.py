from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

class HousingDataCurator:
    def __init__(self):
        self.housing = self.load_housing_data()
        self.train_set = None 
        self.test_set = None
        self._setup_plot_style()

    def load_housing_data(self):
        """Load housing data from source or local file"""
        tarball_path = Path("chapter_2/datasets/housing.tgz")
        if not Path.exists(tarball_path):
            Path.mkdir(Path("chapter_2/datasets"), parents=True, exist_ok=True)
            url = "https://github.com/ageron/data/raw/main/housing.tgz"
            urllib.request.urlretrieve(url, tarball_path)
            with tarfile.open(tarball_path) as housing_tarball:
                housing_tarball.extractall(path="chapter_2/datasets")
        return pd.read_csv(Path("chapter_2/datasets/housing/housing.csv"))

    def display_dataset_info(self):
        """Display basic information about the dataset"""
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', 5)
        
        print("\nFirst 5 rows of the housing dataset:")
        print("=====================================")
        print(self.housing.head())
        
        print("\nDataset info:")
        print("=============")
        print(self.housing.info())
        
        print("\nOcean proximity value counts:")
        print("============================")
        print(self.housing['ocean_proximity'].value_counts())
        
        print("\nDescription of the dataset:")
        print("===========================")
        pd.set_option('display.max_rows', None)
        print(self.housing.describe())

    def _setup_plot_style(self):
        """Set up matplotlib style for visualizations"""
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12

    def plot_feature_distributions(self):
        """Create and display histograms of all features"""
        self.housing.hist(bins=50, figsize=(20,15))
        plt.tight_layout(pad=1.5)
        plt.suptitle('Distribution of Housing Features', fontsize=20, y=1.02)
        plt.savefig('chapter_2/datasets/data_visualization/feature_distributions.png')
        plt.close()

    def plot_income_categories(self):
        """Plot income category distribution"""
        self.housing["income_cat"] = pd.cut(
            self.housing["median_income"],
            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
            labels=[1, 2, 3, 4, 5]
        )
        self.housing["income_cat"].value_counts().sort_index().plot(
            kind="bar", 
            figsize=(10, 7), 
            rot=0, 
            grid=True
        )
        plt.xlabel("Income category")
        plt.ylabel("Number of districts")
        plt.title("Income category counts")
        plt.savefig('chapter_2/datasets/data_visualization/income_categories.png')
        plt.close()

    def split_data_random(self, test_ratio=0.2):
        """Split data using random shuffling"""
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(self.housing))
        test_set_size = int(len(self.housing) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        self.train_set = self.housing.iloc[train_indices]
        self.test_set = self.housing.iloc[test_indices]
        self._print_split_sizes("Random split")

    def _test_set_check(self, identifier, test_ratio):
        """Helper method for hash-based splitting"""
        threshold = int(test_ratio * (2**32))
        hash_value = abs(hash(identifier)) % (2**32)
        return hash_value < threshold

    def split_data_by_id(self, test_ratio=0.2):
        """Split data using identifier-based hashing"""
        self.housing['id'] = self.housing['longitude'] * 1000 + self.housing['latitude']
        ids = self.housing['id']
        in_test_set = ids.apply(lambda id_: self._test_set_check(id_, test_ratio))
        self.train_set = self.housing.loc[~in_test_set]
        self.test_set = self.housing.loc[in_test_set]
        self._print_split_sizes("Hash-based split")

    def split_data_sklearn(self, test_ratio=0.2):
        """Split data using scikit-learn's train_test_split"""
        self.train_set, self.test_set = train_test_split(
            self.housing, test_size=test_ratio, random_state=42
        )
        self._print_split_sizes("Scikit-learn split")

    def split_data_stratified(self, test_ratio=0.2):
        """Split data using stratified sampling based on income categories"""
        if "income_cat" not in self.housing.columns:
            self.housing["income_cat"] = pd.cut(
                self.housing["median_income"],
                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                labels=[1, 2, 3, 4, 5]
            )
        
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
        for train_index, test_index in splitter.split(self.housing, self.housing["income_cat"]):
            self.train_set = self.housing.iloc[train_index]
            self.test_set = self.housing.iloc[test_index]
        
        # Remove income_cat feature after splitting
        for set_ in (self.train_set, self.test_set):
            set_.drop("income_cat", axis=1, inplace=True)
        self.housing.drop("income_cat", axis=1, inplace=True)
        
        self._print_split_sizes("Stratified split")

    def _print_split_sizes(self, split_method):
        """Helper method to print split sizes"""
        print(f"\nDataset split sizes ({split_method}):")
        print("=" * (24 + len(split_method)))
        print(f"Training set size: {len(self.train_set)} samples")
        print(f"Test set size:     {len(self.test_set)} samples")

    def save_train_test_sets(self, prefix="housing"):
        """Save train and test sets to CSV files"""
        if self.train_set is None or self.test_set is None:
            raise ValueError("Must split the data before saving train/test sets")
            
        save_dir = Path("chapter_2/datasets/housing")
        self.train_set.to_csv(save_dir / f"{prefix}_train.csv", index=False)
        self.test_set.to_csv(save_dir / f"{prefix}_test.csv", index=False)
        print(f"\nDatasets saved as '{prefix}_train.csv' and '{prefix}_test.csv'")

    def plot_location_data(self):
        """Plot location data visualizations"""
        # Basic location scatter plot
        self.housing.plot(kind="scatter", x="longitude", y="latitude",
                         grid=True, alpha=0.2)
        plt.savefig('chapter_2/datasets/data_visualization/location_scatter.png')
        plt.close()

        # Location scatter with population density
        self.housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                         s=self.housing["population"] / 100, cmap="jet", colorbar=True,
                         legend=True, sharex=False, figsize=(10, 7))
        plt.savefig('chapter_2/datasets/data_visualization/location_population_density.png')
        plt.close()


if __name__ == "__main__":
    curator = HousingDataCurator()
    curator.display_dataset_info()
    curator.plot_feature_distributions()
    curator.plot_income_categories()
    curator.plot_location_data()
    curator.split_data_stratified()
    curator.save_train_test_sets()
