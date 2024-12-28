import pandas as pd
from chembl_webresource_client.new_client import new_client

# Fetch coronavirus-related targets
target = new_client.target
target_query = target.search('coronavirus')
targets = pd.DataFrame.from_dict(target_query)

# Check which columns are in the DataFrame
print("Available columns in target data:", targets.columns)

# Display available target information based on existing columns
# Adjust columns if 'description' is not present
columns_to_display = ['target_chembl_id', 'pref_name', 'organism', 'target_type']
if 'description' in targets.columns:
    columns_to_display.append('description')

print("\nDrug Target Information:")
print(targets[columns_to_display].head(5))

# Select a specific target for further bioactivity analysis
selected_protein_target = targets['target_chembl_id'][4]  # You can change the index to select different targets

# Retrieve bioactivity data for the selected target
bioactivity_data = new_client.activity
filtered_data = bioactivity_data.filter(target_chembl_id=selected_protein_target).filter(standard_type="IC50")
bioactivity_DF = pd.DataFrame.from_dict(filtered_data)
bioactivity_DF = bioactivity_DF[bioactivity_DF.standard_value.notna()]

# Print a summary of bioactivity data
print(f"\nBioactivity Data for Target: {selected_protein_target}")
print(bioactivity_DF[['molecule_chembl_id', 'canonical_smiles', 'standard_value']].head(10))

# Save raw data for reusability
bioactivity_DF.to_csv('raw_bioactivity_data.csv', index=False)

# Classify activity based on IC50 value thresholds
activity_classes = []
for value in bioactivity_DF.standard_value:
    if float(value) >= 10000:
        activity_classes.append("inactive")
    elif float(value) <= 1000:
        activity_classes.append("active")
    else:
        activity_classes.append("moderate")

bioactivity_DF['activity_class'] = activity_classes

# Print summary of bioactivity classifications
activity_summary = bioactivity_DF['activity_class'].value_counts()
print("\nBioactivity Classification Summary:")
print(activity_summary)

# Optionally, visualize bioactivity distribution per target
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
# Use a single color for the bars
# Update the countplot to assign 'activity_class' to 'hue' and disable the legend
sns.countplot(data=bioactivity_DF, x='activity_class', hue='activity_class', palette="viridis", order=["active", "moderate", "inactive"], legend=False)
plt.title(f"Bioactivity Distribution for Target {selected_protein_target}")
plt.xlabel("Activity Class")
plt.ylabel("Frequency")
plt.show()
