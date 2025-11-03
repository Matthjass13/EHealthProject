# create_server_data.py
import pandas as pd
import os

def create_server_data():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    centers = ['Center-3', 'Center-4', 'Center-5']
    
    # Combine all training data
    train_parts = []
    test_parts = []
    
    for center in centers:
        # Use paths relative to the script location
        train_file = os.path.join(script_dir, "client_data", f"client_{center}_train.csv")
        test_file = os.path.join(script_dir, "client_data", f"client_{center}_test.csv")
        
        train_parts.append(pd.read_csv(train_file))
        test_parts.append(pd.read_csv(test_file))
    
    # Create server data (all centers combined)
    server_train = pd.concat(train_parts, ignore_index=True)
    server_test = pd.concat(test_parts, ignore_index=True)
    
    # Create the folder if it doesn't exist
    health_backup_path = os.path.join(script_dir, "health_data_backup")
    os.makedirs(health_backup_path, exist_ok=True)
    
    # Save the files
    server_train.to_csv(os.path.join(health_backup_path, "train_health_data_server.csv"), index=False)
    server_test.to_csv(os.path.join(health_backup_path, "test_health_data_server.csv"), index=False)
    
    print(f"Created server data: {len(server_train)} train, {len(server_test)} test patients")

if __name__ == "__main__":
    create_server_data()