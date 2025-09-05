import math

# Initial values (you can change these; starting points affect the solution found)
x = 1.0
y = 1.0
a = -10.0  # Start with a > 0 to avoid issues with negative bases in exponents
b = 1.0

# Hyperparameters
learning_rate = 0.01  # Adjust if convergence is too slow/fast (e.g., try 0.001 or 0.1)
num_iterations = 100  # More iterations for better accuracy
target = 1.23
epsilon = 1e-8  # Small value to prevent log(0) or negative bases

# Optimization loop
for i in range(num_iterations):
    # Ensure a is positive for safe exponentiation and log
    # a = max(a, epsilon)
    
    # Compute z
    z = x * y + a ** b
    
    # Compute loss
    loss = (z - target) ** 2
    
    # Compute gradients
    dz = 2 * (z - target)  # Common factor: ∂loss/∂z
    grad_x = dz * y
    grad_y = dz * x
    grad_a = dz * (b * a ** (b - 1)) if b != 0 else 0  # Handle b=0 edge case
    grad_b = dz * (a ** b * math.log(a)) if a > 0 else 0  # Requires a > 0
    
    # Update parameters
    x -= learning_rate * grad_x
    y -= learning_rate * grad_y
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    
    # Optional: Print progress every 1000 iterations
    # if i % 10 == 0:
    print(f"Iteration {i}: loss = {loss:.6f}, z = {z:.6f}, x = {x:.4f}, y = {y:.4f}, a = {a:.4f}, b = {b:.4f}")

# Final results
final_z = x * y + a ** b
print(f"\nFinal x: {x:.4f}")
print(f"Final y: {y:.4f}")
print(f"Final a: {a:.4f}")
print(f"Final b: {b:.4f}")
print(f"Final z: {final_z:.4f}")
