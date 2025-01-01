# genaapl
experimenting with AI stuff.

python -m venv genaapl
genaapl\Scripts\activate

python train.py
python predict.py

pip install transformers
# make sure you download the CUDA compiled version of pytorch by using this arg
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas scikit-learn


https://www.kaggle.com/datasets/deltatrup/aapl-1-minute-historical-stock-data-2006-2024
Date,Open,High,Low,Close,Adj Close,Volume

https://chatgpt.com/share/6773360b-c84c-800c-a11a-c26a73b052f5