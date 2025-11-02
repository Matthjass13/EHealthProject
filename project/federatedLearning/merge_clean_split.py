# merge_clean_split.py
import pandas as pd
import numpy as np
import os

def clean_and_split_data():
    # Since your script is in project/federatedLearning/
    # and dataset is in project/dataset/
    base_path = "C:/VSCode_Projects/E-Health/E-Health_Project/EHealthProject/project/dataset/"  # Go from federatedLearning to project, then into dataset
    
    print(f"Looking for dataset at: {base_path}")
    print(f"Absolute path: {os.path.abspath(base_path)}")
    
    # Check if dataset exists
    if not os.path.exists(base_path):
        print(f"ERROR: Dataset folder not found at {base_path}")
        print("Please make sure your dataset folder is in the project folder")
        return
    
    print(f"✓ Found dataset folder!")
    print(f"Files in dataset folder: {os.listdir(base_path)}")
    
    # Build file paths
    clinical_train_path = os.path.join(base_path, "data_hn_clinical_train.csv")
    clinical_test_path = os.path.join(base_path, "data_hn_clinical_test.csv")
    pt_train_path = os.path.join(base_path, "data_hn_pt_train.csv")
    pt_test_path = os.path.join(base_path, "data_hn_pt_test.csv")
    ct_train_path = os.path.join(base_path, "data_hn_ct_train.csv")
    ct_test_path = os.path.join(base_path, "data_hn_ct_test.csv")
    
    # Check if all files exist
    required_files = [clinical_train_path, clinical_test_path, pt_train_path, pt_test_path, ct_train_path, ct_test_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: Missing file {file_path}")
            return
        else:
            print(f"✓ Found: {os.path.basename(file_path)}")
    
    # Now load the files
    print("\nLoading data...")
    train_merged = pd.read_csv(clinical_train_path).merge(
        pd.read_csv(pt_train_path), on=['PatientID', 'CenterID', 'Outcome'], how='inner'
    ).merge(
        pd.read_csv(ct_train_path), on=['PatientID', 'CenterID', 'Outcome'], how='inner'
    )

    test_merged = pd.read_csv(clinical_test_path).merge(
        pd.read_csv(pt_test_path), on=['PatientID', 'CenterID', 'Outcome'], how='inner'
    ).merge(
        pd.read_csv(ct_test_path), on=['PatientID', 'CenterID', 'Outcome'], how='inner'
    )

    print(f"Before cleaning: {len(train_merged)} train, {len(test_merged)} test patients")
    
    # --- DATA CLEANING ---
    
    # 1. Check for duplicates
    train_duplicates = train_merged.duplicated().sum()
    test_duplicates = test_merged.duplicated().sum()
    print(f"Found {train_duplicates} duplicate patients in train, {test_duplicates} in test")
    
    # Remove duplicates
    train_merged = train_merged.drop_duplicates()
    test_merged = test_merged.drop_duplicates()
    
    # 2. Check for missing values
    print("\nMissing values in training data:")
    missing_train = train_merged.isnull().sum()
    print(missing_train[missing_train > 0])
    
    print("\nMissing values in testing data:")
    missing_test = test_merged.isnull().sum()
    print(missing_test[missing_test > 0])
    
    # 3. Handle missing values - simple approach
    # For numerical columns, fill with median
    numerical_cols = train_merged.select_dtypes(include=[np.number]).columns
    train_merged[numerical_cols] = train_merged[numerical_cols].fillna(train_merged[numerical_cols].median())
    test_merged[numerical_cols] = test_merged[numerical_cols].fillna(test_merged[numerical_cols].median())
    
    # 4. Check for constant columns (that don't provide information)
    constant_cols = []
    for col in train_merged.columns:
        if train_merged[col].nunique() == 1:  # Only one unique value
            constant_cols.append(col)
    
    if constant_cols:
        print(f"Removing constant columns: {constant_cols}")
        train_merged = train_merged.drop(columns=constant_cols)
        test_merged = test_merged.drop(columns=constant_cols)
    
    print(f"After cleaning: {len(train_merged)} train, {len(test_merged)} test patients")
    
    # --- SPLIT BY CENTERS ---
    
    # Create client_data folder in the SAME FOLDER as this script (federatedLearning)
    client_data_path = "C:/VSCode_Projects/E-Health/E-Health_Project/EHealthProject/project/federatedLearning/client_data"  # This creates it in the federatedLearning folder
    os.makedirs(client_data_path, exist_ok=True)
    print(f"\nCreated client_data folder at: {os.path.abspath(client_data_path)}")
    
    # Find all unique centers in your data
    centers = train_merged['CenterID'].unique()
    print(f"Found centers: {centers}")
    
    # Split and save CSV for each center
    for center in centers:
        # Training data for this center
        center_train = train_merged[train_merged['CenterID'] == center]
        center_train_path = os.path.join(client_data_path, f"client_{center}_train.csv")
        center_train.to_csv(center_train_path, index=False)
        
        # Testing data for this center
        center_test = test_merged[test_merged['CenterID'] == center]
        center_test_path = os.path.join(client_data_path, f"client_{center}_test.csv")
        center_test.to_csv(center_test_path, index=False)
        
        print(f"Created files for {center}: {len(center_train)} train, {len(center_test)} test samples")
        print(f"  → {center_train_path}")
        print(f"  → {center_test_path}")
    
    print(f"\n✓ All client CSV files created in: {os.path.abspath(client_data_path)}")
    print("You can now use these files in your FL notebook!")

# Run the function
clean_and_split_data()