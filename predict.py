import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

NUM_PREDICTION_VALUE = 180

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

# for name, param in model.named_parameters():
#     print(name, param.data)

# Load and preprocess the data
data = pd.read_csv('very_small.csv')
#data = pd.read_csv('test_data.csv')
data = data.drop('Date', axis=1)  # Drop unnecessary columns

# Load the saved scaler
#scaler = joblib.load('scaler.pkl')
scaler = MinMaxScaler()

# Normalize the data
scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaled_data

# print("Normalized input data:")
# print(scaled_data[:5])  # Print the first few rows of normalized data

# dummy_input = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32).unsqueeze(0).to(device)
# print("Model output for identical input:")
# print(model(dummy_input))

# Prepare the initial historical data
X = data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]  # Match training features
historical_features = torch.tensor(X.values, dtype=torch.float32).to(device)  # Entire dataset as input

# Predict 10 new "Close" values based on historical data
predictions = []
with torch.no_grad():
    for _ in range(NUM_PREDICTION_VALUE):  # Generate new predictions
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
# print("Next predicted 'Close' values:")
# print(predictions)


# Combine original and predicted data
original_close = scaler.inverse_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])[:, 3]  # Extract original 'Close'
combined_close = list(original_close) + predictions  # Append predicted values

# Plot the data
plt.figure(figsize=(12, 6))

# Plot original 'Close' values
plt.plot(range(len(original_close)), original_close, label='Original Close', color='blue')

# Determine the overall trend of predictions
overall_color = 'green' if predictions[-1] > predictions[0] else 'red'

# Plot predicted values as one line with a single color
plt.plot(
    range(len(original_close), len(original_close) + len(predictions)),
    predictions,
    color=overall_color,
    linestyle='--',
    label=f'Predicted Close ({ "Ascending" if overall_color == "green" else "Descending" })'
)

# Final plot adjustments
plt.title('Original and Predicted Close Prices')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.savefig('plot.png')
