import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from transformers import TransformerModel, TransformerConfig

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import joblib

print(f'Step 1: use the GPU')
# ===================================
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.current_device())  # Should print the current GPU device ID
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Should print the GPU name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(f'Step 2: Preprocess the Data')
# ============================================
def read_data(file):
    # Load the CSV file
    data = pd.read_csv(file)

    # Drop the original 'Date' column if not needed
    data = data.drop('Date', axis=1)

    # Normalize numerical features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
    data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaled_data

    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')

    # split into features and targets
    X = data[['Open', 'High', 'Low', 'Adj Close', 'Volume']] # features
    y = data['Close']  # Target
    return X,y

#X,y = read_data('large.csv')
X,y = read_data('small.csv')
#X,y = read_data('very_small.csv')

print(f'Step 3: Create a Dataset Class')
# ============================================


class StockDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

dataset = StockDataset(X, y)

print(f'Step 4: Define the Model')
# =========================================

class StockTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(StockTransformer, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=2,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.input_layer(x)
        # print(f"After input layer: {x.shape}")
        x = x.unsqueeze(1)
        # print(f"After adding sequence dimension: {x.shape}")
        x = self.transformer(x, x)
        # print(f"After transformer: {x.shape}")
        x = self.output_layer(x[:, -1])
        # print(f"After output layer: {x.shape}")
        return x

print(f'input_size={X.shape[1]}')
model = StockTransformer(input_size=X.shape[1], hidden_size=128, num_layers=2).to(device)
print(next(model.parameters()).device)  # Should print 'cuda:0'
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

print(f'Step 5: Train the Model')
# =============================

# Define the dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
model.train()
start_time = time.time()  # Record the start time of training
log_interval = 60  # Log every 60 seconds (1 minute)

for epoch in range(num_epochs):
    epoch_start_time = time.time()  # Record the start time of the epoch
    epoch_loss = 0
    batch_start_time = time.time()  # Track time for logging

    for i, (features, targets) in enumerate(dataloader):
        # Move data to GPU
        features, targets = features.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions.squeeze(), targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Log progress every minute
        if time.time() - batch_start_time > log_interval:
            elapsed_time = time.time() - start_time
            print(f"[Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(dataloader)}] "
                  f"Elapsed Time: {elapsed_time:.2f}s, Current Loss: {loss.item():.4f}")
            batch_start_time = time.time()  # Reset log timer

    # Calculate epoch time
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, Loss: {epoch_loss:.4f}")

    # Estimate remaining time
    elapsed_time = time.time() - start_time
    estimated_total_time = (elapsed_time / (epoch + 1)) * num_epochs
    estimated_completion_time = start_time + estimated_total_time
    remaining_time = estimated_total_time - elapsed_time

    print(f"Estimated Completion Time: {time.ctime(estimated_completion_time)} "
          f"(Remaining Time: {remaining_time / 60:.2f} minutes)")

print(f'Step 6: Evaluate the Model')
# ===============================
# Assuming you have a test dataset
X_test,y_test = read_data('test_data.csv')
test_dataset = StockDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=32)

model.eval()
test_loss = 0
with torch.no_grad():
    for features, targets in test_dataloader:
        features, targets = features.to(device), targets.to(device)
        predictions = model(features)
        loss = criterion(predictions.squeeze(), targets)
        test_loss += loss.item()
print(f"Test Loss: {test_loss:.4f}")


print(f"Features shape: {features.shape}, Targets shape: {targets.shape}")
print(f"Sample feature: {features[0]}, Target: {targets[0]}")

print(f'Step 7: Save the Model')
# ====================================
torch.save(model.state_dict(), 'stock_transformer_model.pth')
# To load the model
#model.load_state_dict(torch.load('stock_transformer_model.pth'))

