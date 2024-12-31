import torch
from datetime import datetime

print(datetime.now().date())
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.current_device())  # Should print the current GPU device ID
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Should print the GPU name
print(datetime.now().date())
