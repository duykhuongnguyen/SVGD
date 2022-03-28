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
output = model(test_data)[0, 0].item()
print(f'Prediction: {output * 100:.2f}%\n')
output = output > 0.5

from svgd import SVGD
attacker = SVGD(model, num_steps=100, epsilon=1, step_size=1/255, clip_min=-1)

def postprocess(x):
    pos = torch.max(x[:, 3:], dim=1)[1]
    x = x.clone()
    x[:, 3:] = 0
    for i, j in enumerate(pos):
        x[i, 3 + j] = 1
    return x

x_adv_list = attacker(test_data, ensemble=False, postprocess=postprocess, output_all=True)

found = [False] * 4
x_adv = torch.empty_like(x_adv_list[0])
for x_adv_ in x_adv_list:
    x_adv_ = postprocess(x_adv_)
    output_ = model(x_adv_)
    for i in range(4):
        if not found[i]:
            if (output_[i, 0].item() > 0.5) != output:
                x_adv[i, ...] = x_adv_[i, ...]
                found[i] = True
    if all(found):
        break

count = 0
for i in range(4):
    if found[i]:
        count += 1
        print(f'[+] Counterfactual #{count}')
        print_data(x_adv[i, ...])
        print(f'Prediction: {model(x_adv)[i, 0].item() * 100:.2f}%\n')
