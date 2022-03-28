import pickle
import torch


# Load data
with open('results/test_data.pkl', 'rb') as f:
    X_test = pickle.load(f)
    X_test = torch.tensor(X_test, dtype=torch.float32)

# Load model
with open('checkpoints/net0_german_0.pickle', 'rb') as f:
    model = pickle.load(f)
