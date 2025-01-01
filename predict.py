import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

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
        self.output_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.transformer(x, x)
        return self.output_layer(x[:, -1])  # Use the last timestep's output

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StockTransformer(input_size=5, hidden_size=128, num_layers=2)  # Match input_size to training
model.load_state_dict(torch.load('stock_transformer_model.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()  # Always put into eval mode during inference

# Load and preprocess the data
data = pd.read_csv('test_data.csv')
data = data.drop('Date', axis=1)  # Drop unnecessary columns

# Load the saved scaler
scaler = joblib.load('scaler.pkl')

# Normalize the data
scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaled_data

# Prepare the initial historical data
X = data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]  # Match training features
historical_features = torch.tensor(X.values, dtype=torch.float32).to(device)  # Entire dataset as input

# Predict 10 new "Close" values based on historical data
predictions = []
with torch.no_grad():
    for _ in range(10):  # Generate 10 new predictions
        # Add batch dimension for input
        input_features = historical_features.unsqueeze(0)  # Use entire historical sequence
        predicted_close = model(input_features).item()

        # Prepare data for inverse transformation
        input_for_inverse = [[0, 0, 0, predicted_close, 0, 0]]  # Place in 'Close' position

        # Perform inverse transformation
        original_scale_value = scaler.inverse_transform(input_for_inverse)[0, 3]  # Extract 'Close'
        predictions.append(original_scale_value)

        # Simulate appending the new prediction to historical features
        new_row = torch.tensor([[0, 0, 0, predicted_close, 0]], dtype=torch.float32).to(device)  # New prediction as tensor
        historical_features = torch.cat([historical_features, new_row], dim=0)  # Append to historical features

# Print the predictions
print("Next 10 predicted 'Close' values:")
print(predictions)

# Optionally, append predictions to the DataFrame and save to a new CSV
new_data = pd.DataFrame({'Close': predictions})
new_data.to_csv('predicted_data.csv', index=False)
print("Predictions saved to 'predicted_data.csv'")
