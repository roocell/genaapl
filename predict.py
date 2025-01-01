import torch
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
        self.output_layer = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x, x)
        return self.output_layer(x[:, -1])  # Use the last timestep's output

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StockTransformer(input_size=5, hidden_size=128, num_layers=2)  # Match input_size to training
model.load_state_dict(torch.load('stock_transformer_model.pth', weights_only=True))
model.to(device)
model.eval() # always put into eval when doing inference (using the model)

# Load and preprocess the data
data = pd.read_csv('test_data.csv')
data = data.drop('Date', axis=1)

# Normalize the data
#scaler = MinMaxScaler()
scaler = joblib.load('scaler.pkl') # use saved scaler

scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaled_data

# Prepare the most recent data as the input
X = data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]  # Match training features
last_features = torch.tensor(X.values[-1], dtype=torch.float32).to(device)  # Last row as input


print(f'{data}')

# Predict the next 10 "Close" values
predictions = []
with torch.no_grad():
    for _ in range(10):
        # Add batch dimension
        input_features = last_features.unsqueeze(0)
        predicted_close = model(input_features).item()

        # Prepare data for inverse transformation
        # Insert the predicted value into the correct position based on the scaler's expected feature order
        input_for_inverse = [[0, 0, 0, predicted_close, 0, 0]]  # Example for ['Open', 'High', 'Low', 'Adj Close', 'Volume']
        
        # Perform inverse transformation
        original_scale_value = scaler.inverse_transform(input_for_inverse)[0, 3]  # Extract the 'Close' column
        
        # Print the inverse-transformed value
        print(f'Inverse-transformed predicted close: {original_scale_value}')

        predictions.append(original_scale_value)

        # topk?

        # Simulate updating the last_features with predicted value
        new_row = torch.tensor(last_features.tolist()[:3] + [predicted_close] + last_features.tolist()[4:], dtype=torch.float32).to(device)
        last_features = new_row


# Print the results
print("Next 10 predicted 'Open' values:")
print(predictions)
