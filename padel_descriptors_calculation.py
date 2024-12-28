import pandas as pd
import numpy as np
import seaborn as sns
import glob
from padelpy import padeldescriptor
import zipfile

# Specify the path to the uploaded file (e.g., fingerprints_xml.zip)
zip_path = "fingerprints_xml.zip"
# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("extracted_data")  # Choose a directory to extract to

print("Files extracted successfully.")

# Get absolute paths to xml files
xml_files = glob.glob("extracted_data/*.xml")
xml_files.sort()
print(f"XML files found: {xml_files}")

# Import the normalized activity data
bioactivity_DF = pd.read_csv('normalized_data.csv')
# print("Bioactivity DataFrame loaded:")
# print(bioactivity_DF.head(5))

filtered_DF = bioactivity_DF[['canonical_smiles', 'molecule_chembl_id']]
# print("Filtered DataFrame (for SMILES and IDs):")
# print(filtered_DF.head(10))

# Export data to csv format for padel readability
filtered_DF.to_csv('molecule.smi', sep='\t', index=False, header=False)
print("SMILES file 'molecule.smi' created for PaDEL input.")

# Molecular fingerprints are sorted and so are their respective xml files
fingerprints = [
    'AtomPairs2DCount', 'AtomPairs2D', 'EState', 'CDKextended', 'CDK',
    'CDKgraphonly', 'KlekotaRothCount', 'KlekotaRoth', 'MACCS',
    'PubChem', 'SubstructureCount', 'Substructure'
]
FP_hashmap = dict(zip(fingerprints, xml_files))
print("Fingerprint-XML mapping created.")

# Select fingerprint (Substructure)
fingerprint = 'Substructure'
fingerprint_output_file = ''.join([fingerprint, '.csv'])  # Substructure.csv
fingerprint_descriptortypes = FP_hashmap[fingerprint]
print(f"Using fingerprint '{fingerprint}' with XML descriptor: {fingerprint_descriptortypes}")

# Generate descriptors using PaDEL
padeldescriptor(
    mol_dir='molecule.smi',
    d_file=fingerprint_output_file,
    descriptortypes=fingerprint_descriptortypes,
    detectaromaticity=True,
    standardizenitro=True,
    standardizetautomers=True,
    threads=2,
    removesalt=True,
    log=True,
    fingerprints=True
)

print(f"Fingerprint descriptors saved to {fingerprint_output_file}")

# Load and transform the feature DataFrame
features_DF = pd.read_csv(fingerprint_output_file)
print("Feature DataFrame loaded:")
print(features_DF.head(5))

# Drop the 'Name' column
features_DF = features_DF.drop(columns=['Name'])
# print("Features DataFrame after dropping 'Name' column:")
# print(features_DF.head(5))

# Set targets for pIC50 and combine for regression model
targets = bioactivity_DF['pIC50']
model_input_DF = pd.concat([features_DF, targets], axis=1)
print("Regression model input data:")
print(model_input_DF.head(10))

# Export regression data
model_input_DF.to_csv('regression_model_data.csv', index=False)
print("Regression model data saved.")

# Prepare classification data by dropping pIC50 and adding activity_class
model_input_DF.drop(columns=['pIC50'], inplace=True)
targets = bioactivity_DF['activity_class']
model_input_DF = pd.concat([features_DF, targets], axis=1)
print("Classification model input data:")
print(model_input_DF.head(5))

# Export classification data
model_input_DF.to_csv('classification_model_data.csv', index=False)
print("Classification model data saved.")
