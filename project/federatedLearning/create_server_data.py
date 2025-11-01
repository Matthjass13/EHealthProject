# create_server_data.py
import pandas as pd
import os

def create_server_data():
    centers = ['Center-3', 'Center-4', 'Center-5']
    
    # Combine all training data
    train_parts = []
    test_parts = []
    
    for center in centers:
        train_file = f"C:/VSCode_Projects/E-Health/E-Health_Project/EHealthProject/project/federatedLearning/client_data/client_{center}_train.csv"
        test_file = f"C:/VSCode_Projects/E-Health/E-Health_Project/EHealthProject/project/federatedLearning/client_data/client_{center}_test.csv"
        
        train_parts.append(pd.read_csv(train_file))
        test_parts.append(pd.read_csv(test_file))
    
    # Create server data (all centers combined)
    server_train = pd.concat(train_parts, ignore_index=True)
    server_test = pd.concat(test_parts, ignore_index=True)
    
    # Create the folder if it doesn't exist
    os.makedirs("C:/VSCode_Projects/E-Health/E-Health_Project/EHealthProject/project/federatedLearning/health_data_backup", exist_ok=True)
    
    # Save the files
    server_train.to_csv("C:/VSCode_Projects/E-Health/E-Health_Project/EHealthProject/project/federatedLearning/health_data_backup/train_health_data_server.csv", index=False)
    server_test.to_csv("C:/VSCode_Projects/E-Health/E-Health_Project/EHealthProject/project/federatedLearning/health_data_backup/test_health_data_server.csv", index=False)
    
    print(f"Created server data: {len(server_train)} train, {len(server_test)} test patients")

create_server_data()