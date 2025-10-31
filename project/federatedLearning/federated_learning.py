import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from copy import deepcopy

# Model definition
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training and evaluation loops
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    for features, labels in loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def load_dataset(train_file, test_file):
    # Load data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    class_names = train_data["Class"].unique()
    print(f"Classes: {class_names}")
    # Feature and label columns
    feature_names = train_data.columns[:-1]
    print(f"Features: {feature_names}")
    X_train = train_data[feature_names].values
    y_train = train_data["Class"].values
    X_test = test_data[feature_names].values
    y_test = test_data["Class"].values

    # Convert labels to integers
    class_map = {label: idx for idx, label in enumerate(class_names)}
    y_train = [class_map[label] for label in y_train]
    y_test = [class_map[label] for label in y_test]

    # Split training data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")

    input_dim = X_train.shape[1]
    output_dim = len(class_names)

    return X_train, y_train, X_val, y_val, X_test, y_test, input_dim, output_dim

def features_scaling(X_train, X_val, X_test):
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled successfully.")
    return X_train_scaled, X_val_scaled, X_test_scaled


def convert_to_tensors(X_train, y_train, X_val, y_val, X_test, y_test):
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor

def create_dataloader(X_tensor, y_tensor, batch_size):
    # Create DataLoader objects
    data = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return loader

HIDDEN_DIM = 64
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATIENCE = 5  # Early stopping patience

# Load and preprocess server data
X_train_server, y_train_server, X_val_server, y_val_server, X_test_server, y_test_server, INPUT_DIM, OUTPUT_DIM = load_dataset(
    "project/dataset/data_hn_clinical_train.csv", "project/dataset/data_hn_clinical_test.csv")

X_train_server, X_val_server, X_test_server = features_scaling(X_train_server, X_val_server, X_test_server)

X_train_tensor_server, y_train_tensor_server, X_val_tensor_server, y_val_tensor_server, X_test_tensor_server, y_test_tensor_server = convert_to_tensors(X_train_server, y_train_server, X_val_server, y_val_server, X_test_server, y_test_server)

train_loader_server = create_dataloader(X_train_tensor_server, y_train_tensor_server, BATCH_SIZE)
val_loader_server = create_dataloader(X_val_tensor_server, y_val_tensor_server, BATCH_SIZE)
test_loader_server = create_dataloader(X_test_tensor_server, y_test_tensor_server, BATCH_SIZE)

print(f"Device: {DEVICE}, Input dim: {INPUT_DIM}, Output dim: {OUTPUT_DIM}")

# Train server

# Model, optimizer, and loss
server_model = MLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM).to(DEVICE)
optimizer = torch.optim.Adam(server_model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Early stopping setup
best_val_loss = float('inf')
patience_counter = 0


# Training loop with early stopping
print("Starting training...")
for epoch in range(EPOCHS):
    train_loss, train_acc = train(server_model, train_loader_server, optimizer, criterion)
    val_loss, val_acc = evaluate(server_model, val_loader_server, criterion)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        print("New best validation loss. Resetting patience counter.")
    else:
        patience_counter += 1
        print(f"No improvement. Patience counter: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Test the model
test_loss, test_acc = evaluate(server_model, test_loader_server, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Collect predictions for metrics
server_model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for features, labels in test_loader_server:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        outputs = server_model(features)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Confusion matrix and other metrics
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# FL parameters
aggregation_freq = 5
aggregation_round = round(EPOCHS / aggregation_freq)

print(f"Aggregation_round: {aggregation_round}")
clients_number = 5

# FL clients data
train_loader_clients = []
val_loader_clients = []
test_loader_clients = []

models = []

def fed_avg(models_weights):
    # Average the weights
    new_state_dict = {}
    for key in models_weights[0].keys():
        new_state_dict[key] = sum([m[key] for m in models_weights]) / len(models_weights)
    return new_state_dict

##def fed_avg_weighted(<parameters>):

global_model = deepcopy(server_model)

# Load and preprocess clients data

# Create models for each client with server weights