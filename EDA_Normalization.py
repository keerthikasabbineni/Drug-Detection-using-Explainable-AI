from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random import seed
from scipy.stats import mannwhitneyu

# Load bioactivity data
bioactivity_DF = pd.read_csv('preprocessed_bioactivity_data.csv')
print("Bioactivity data loaded:")
print(bioactivity_DF.head())

# Function to compute Lipinski descriptors
def lipinski(smiles, verbose=False):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData = np.arange(1, 1)
    i = 0
    for mol in moldata:
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)

        row = np.array([desc_MolWt, desc_MolLogP, desc_NumHDonors, desc_NumHAcceptors])

        if (i == 0):
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        i += 1
    columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)

    return descriptors

# Calculate Lipinski descriptors
lipinski_data = lipinski(bioactivity_DF.canonical_smiles)
print("\nLipinski descriptors calculated:")
print(lipinski_data.head())

# Merge Lipinski descriptors with bioactivity data
final_DF = pd.concat([bioactivity_DF, lipinski_data], axis=1)
print("\nMerged DataFrame shape:", final_DF.shape)
print(final_DF.head())

# Show the original standard_value distribution
print("\nDescription of 'standard_value' before normalization:")
print(bioactivity_DF.standard_value.describe())

# Boxplot of the original data distribution
plt.figure(figsize=(12, 5))
plt.xlim(-100000, 1000000)
sns.boxplot(x=final_DF.standard_value)
plt.show()

# Normalize standard values
norm = [min(i, 100000000) for i in final_DF['standard_value']]
final_DF['pIC50'] = norm
final_DF = final_DF.drop(columns=['standard_value'])

# Updated pIC50 distribution
print("\nUpdated 'pIC50' distribution after normalization:")
print(final_DF.pIC50.describe())

# Visualize normalized data with boxplots
plt.figure(figsize=(12, 5))
plt.xlim(-100000, 1000000)
sns.boxplot(x=final_DF.pIC50)
plt.show()

# Log10 normalization for pIC50
final_DF['pIC50'] = final_DF['pIC50'].apply(lambda x: x * (10 ** -9))
final_DF['pIC50'] = -np.log10(final_DF['pIC50'])
print("\nFinal 'pIC50' values after log normalization:")
print(final_DF.pIC50.head(10))

# Plot normalized values
plt.figure(figsize=(12, 5))
plt.xlim(0, 14)
sns.boxplot(x=final_DF.pIC50)
plt.show()

# Count plot of activity class
plt.figure(figsize=(5.5, 5.5))
sns.countplot(x='activity_class', data=final_DF, edgecolor='black')
plt.xlabel('Activity Class', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.show()

# Removing moderate activity class
final_DF = final_DF[final_DF.activity_class != 'moderate']
print("\nFinal DataFrame after removing 'moderate' class:")
print(final_DF['activity_class'].value_counts())

# Scatter plot of MW vs LogP
plt.figure(figsize=(8, 8))
sns.scatterplot(x='MW', y='LogP', data=final_DF, hue='activity_class', size='pIC50', edgecolor='black', alpha=0.7)
plt.xlabel('MW', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.show()

# Mann-Whitney U test function
def run_mannwhitney(descriptor):
    seed(1)
    active_molecules = final_DF[final_DF.activity_class == 'active'][descriptor]
    inactive_molecules = final_DF[final_DF.activity_class == 'inactive'][descriptor]
    stats, pvalue = mannwhitneyu(active_molecules, inactive_molecules)
    print(f"\nMann-Whitney U test for {descriptor}: Statistics={stats}, p-value={pvalue}")
    results = pd.DataFrame({'Descriptor': descriptor, 'Statistics': stats, 'p-value': pvalue}, index=[0])
    results.to_csv(f'mannwhitneyu_{descriptor}.csv')
    return results

# Create boxplot function
def get_boxplot(descriptor):
    plt.figure(figsize=(8, 8))
    sns.boxplot(x='activity_class', y=descriptor, data=final_DF)
    plt.xlabel('Activity Class', fontsize=14, fontweight='bold')
    plt.ylabel(descriptor, fontsize=14, fontweight='bold')
    plt.show()

# Running analysis on different descriptors
for descriptor in ['pIC50', 'MW', 'LogP', 'NumHDonors', 'NumHAcceptors']:
    get_boxplot(descriptor)
    run_mannwhitney(descriptor)
