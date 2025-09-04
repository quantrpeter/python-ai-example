import torch

# Initialize parameters with some starting values (these can be changed)
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Set up optimizer (Adam is efficient for this)
optimizer = torch.optim.Adam([x, y, a, b], lr=0.1)  # lr is learning rate; adjust if needed

# Target value
target = torch.tensor(1.23)

# Optimization loop
for i in range(1000):  # More iterations can improve accuracy
    z = x * y + a ** b
    loss = (z - target) ** 2
    optimizer.zero_grad()  # Reset gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters

# Get final values
print(f"Final x: {x.item()}")
print(f"Final y: {y.item()}")
print(f"Final a: {a.item()}")
print(f"Final b: {b.item()}")
print(f"Final z: {x.item() * y.item() + a.item() ** b.item()}")

