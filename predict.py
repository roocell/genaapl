import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the same StockTransformer model class
class StockTransformer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(StockTransformer, self).__init__()
        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.transformer = torch.nn.Transformer(
            d_model=hidden_size,
            nhead=4,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x, x)
        return self.output_layer(x[:, -1])  # Use the last timestep's output

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StockTransformer(input_size=8, hidden_size=128, num_layers=2)  # Match input_size to training
model.load_state_dict(torch.load('stock_transformer_model.pth', weights_only=True))
model.to(device)
model.eval()

# Load and preprocess the data
data = pd.read_csv('test_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data = data.drop('Date', axis=1)

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['High', 'Low', 'Close', 'Adj Close', 'Volume', 'Year', 'Month', 'Day']])
data[['High', 'Low', 'Close', 'Adj Close', 'Volume', 'Year', 'Month', 'Day']] = scaled_data

# Prepare the most recent data as the input
X = data[['High', 'Low', 'Close', 'Adj Close', 'Volume', 'Year', 'Month', 'Day']]  # Match training features
last_features = torch.tensor(X.values[-1], dtype=torch.float32).to(device)  # Last row as input

# Predict the next 10 "Open" values
predictions = []
with torch.no_grad():
    for _ in range(10):
        # Add batch dimension
        input_features = last_features.unsqueeze(0)
        predicted_open = model(input_features).item()
        predictions.append(predicted_open)

        # Simulate updating the last_features with predicted value
        new_row = torch.tensor([predicted_open] + last_features.tolist()[1:], dtype=torch.float32).to(device)
        last_features = new_row

# Convert predictions back to original scale
predicted_open_values = scaler.inverse_transform(
    [[0, 0, 0, 0, 0, 0, 0, p] for p in predictions]
)[:, -1]

# Print the results
print("Next 10 predicted 'Open' values:")
print(predicted_open_values)
