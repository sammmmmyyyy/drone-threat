import pandas as pd
import re

# File Paths
drone_file = "C:\\Users\\dream\\Documents\\AI_drone\\datasets\\drone_dataset\\drone_classified.csv"
rf_file = "C:\\Users\\dream\\Documents\\AI_drone\\datasets\\drone_dataset\\drone_rf_values.csv"
classified_file = "C:\\Users\\dream\\Documents\\AI_drone\\datasets\\drone_dataset\\drone_rf_classified.csv"

# Load datasets
drone_df = pd.read_csv(drone_file)
rf_df = pd.read_csv(rf_file)

# Debugging: Print column names and sample values
print("\U0001F4CC Drone Data Columns:", drone_df.columns)
print("\U0001F4CC RF Data Columns:", rf_df.columns)

# Extract numerical part of Drone_ID in rf_df and store it in a new 'seq' column
rf_df['seq'] = rf_df['Drone_ID'].apply(lambda x: int(re.search(r'\d+', str(x)).group()) if pd.notna(x) and re.search(r'\d+', str(x)) else None)

# Check unique IDs
print("\U0001F539 Unique seq values in Drone Dataset:", drone_df['seq'].unique())
print("\U0001F539 Unique seq values in RF Dataset:", rf_df['seq'].unique())

# Attempt to merge based on 'seq'
merged_df = pd.merge(drone_df, rf_df.drop(columns=['Drone_ID']), on="seq", how="left")

# Check if merge was successful
if merged_df['seq'].isnull().all():
    print("❌ Merging failed! No matching seq values found. Verify that seq exists in both datasets.")
else:
    print("✅ Merging successful! Saving data...")
    merged_df.to_csv(classified_file, index=False)
    print(f"RF-integrated dataset saved to: {classified_file}")
