from microjax import grad, exp
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + exp(-x))


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


def forward(func, vec):
    return [func(x) for x in vec]


x = [i / 100 for i in range(-500, 500)]

primals = forward(tanh, x)

# first derivative
f_prime = forward(grad(tanh), x)

# # higher order derivatives
f_double_prime = forward(grad(grad(tanh)), x)
f_triple_prime = forward(grad(grad(grad(tanh))), x)
f_fourth_prime = forward(grad(grad(grad(grad(tanh)))), x)


# Plotting with high resolution
plt.figure(figsize=(12, 8))
plt.style.use("dark_background")  # Set the style to dark background

plt.plot(x, primals, label="f(x) = tanh(x)", linewidth=2)
plt.plot(x, f_prime, label="f'(x)", linewidth=2)
plt.plot(x, f_double_prime, label="f''(x)", linewidth=2)
plt.plot(x, f_triple_prime, label="f'''(x)", linewidth=2)
plt.plot(x, f_fourth_prime, label="f''''(x)", linewidth=2)

plt.title("Plot of tanh(x) and its First Four Derivatives", fontsize=22, weight="bold")
plt.xlabel("x", fontsize=18)
plt.ylabel("Function value", fontsize=18)
plt.legend(fontsize=16)
plt.grid(True, linestyle="--", alpha=0.15)  # Reduced grid visibility
plt.tight_layout()

plt.box(False)

plt.savefig("grad_plot.png")
plt.show()
