import pickle
import torch


# Load data
with open('results/test_data.pkl', 'rb') as f:
    X_test = pickle.load(f)
    X_test = torch.tensor(X_test, dtype=torch.float32)

# Load model
with open('checkpoints/net0_german_0.pickle', 'rb') as f:
    model = pickle.load(f)

class Expand(torch.nn.Module):
    def forward(self, x):
        return torch.hstack([x, 1-x])

model = torch.nn.Sequential(model, Expand())

def print_data(data):
    print(f'Duration: {data[0]:.4f}')
    print(f'Amount: {data[1]:.4f}')
    print(f'Age: {data[2]:.4f}')
    print(f'PSS: {data[3:]}')

# NOTE: SELECT TEST DATA HERE
test_data = X_test[42:43, ...]
print('[+] Clean example')
print_data(test_data[0, ...])
print(f'Prediction: {model(test_data)[0, 0].item() * 100:.2f}%\n')

from svgd import SVGD
attacker = SVGD(model)

x_adv = attacker(test_data, ensemble=False)
for i in range(4):
    print(f'[+] Counterfactual #{i}')
    print_data(x_adv[i, ...])
    print(f'Prediction: {model(x_adv)[i, 0].item() * 100:.2f}%\n')
