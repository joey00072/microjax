from microjax import grad, exp, sin, value_and_grad

import numbers
import random
import math


class Module:
    """
    i am sure there better ways to do this,
    but its 4am, coffine in my system stating to loose its effect,
    i can write todo: but i am not gonna do it later, so.. it is what it
    """

    def state_dict(self):
        def _state_dict(root, state):
            state = {}
            for k, v in root.__dict__.items():
                if isinstance(v, numbers.Number):
                    state[k] = v
                elif isinstance(v, ModuleList):
                    sub_state = [
                        (m.state_dict() if isinstance(m, Module) else m) for m in v
                    ]
                    for i, item in enumerate(sub_state):
                        if isinstance(item, dict):
                            for sub_k, sub_v in item.items():
                                state[f"{k}.{i}.{sub_k}"] = sub_v
                        else:
                            state[f"{k}.{i}"] = item
                elif isinstance(v, Module):
                    sub_state = v.state_dict()
                    for sub_k, sub_v in sub_state.items():
                        state[f"{k}.{sub_k}"] = sub_v
            return state

        return _state_dict(self, {})

    def load_state_dict(self, state):
        sub_state = {}
        for key, value in state.items():
            if "." in key:
                attrs_name = key.split(".")
                attr = self
                for name in attrs_name[:-1]:
                    if isinstance(attr, ModuleList):
                        attr = attr[int(name)]
                    else:
                        attr = getattr(attr, name)

                if isinstance(attr, ModuleList):
                    attr[int(attrs_name[-1])] = value
                else:
                    setattr(attr, attrs_name[-1], value)
            else:
                setattr(self, key, value)


class ModuleList(Module, list):
    def __init__(self, args):
        super().__init__(args)


class Neuron(Module):
    def __init__(self, n_inputs):
        self.weights = ModuleList(
            [(random.randint(-100, 100) / 100) for _ in range(n_inputs)]
        )
        self.bias = 0.0

    def __call__(self, inputs):
        output = 0
        for w, x in zip(self.weights, inputs):
            output = output + w * x
        return output + self.bias


class Layer(Module):
    def __init__(self, n_inputs, n_outputs):
        self.layers = ModuleList([Neuron(n_inputs) for _ in range(n_outputs)])

    def __call__(self, inputs):
        # print(f"self.layers: {self.layers}")
        return [layer(inputs) for layer in self.layers]


# ===========
def sigmoid(x):
    return 1 / (1 + exp(-x))


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


class Sigmoid(Module):
    def __call__(self, inputs):
        return [sigmoid(x) for x in inputs]


class Tanh(Module):
    def __call__(self, inputs):
        return [tanh(x) for x in inputs]


# ===========

# when you do gfunc = grad(func) it makes closure over func
# all args to gfunc(args) are boxed to calculate gradients
# that why we can pass model model directly
# with this func_with_model_state model_forward will take state as input
# but will act as if it recived model


def func_with_model_state(func, model):
    def model_forward(state, *args):
        model.load_state_dict(state)
        return func(model, *args)

    return model_forward


# ============


class MLP(Module):
    def __init__(self):
        self.l1 = Layer(2, 4)
        self.l2 = Layer(4, 1)
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()

    def __call__(self, x):
        x = self.l1(x)
        x = self.tanh(x)
        x = self.l2(x)
        x = self.sigmoid(x)
        return x


def train_step(model, inputs, targets):
    loss = 0
    for x, y in zip(inputs, targets):
        preds = 0
        preds = model(x)
        local_loss = 0

        # <mse_loss>
        for p, t in zip(preds, y):
            diff = p - t
            local_loss += diff * diff
        loss += local_loss / len(preds)
        # </mse_loss>

    return loss / len(inputs)


class AdamOptimizer:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, weights: dict, gradients: dict):
        for key in gradients:
            if key not in self.m:
                self.m[key] = 0
                self.v[key] = 0

        self.t += 1

        for key, grad in gradients.items():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad**2)
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            weights[key] -= self.learning_rate * m_hat / (v_hat**0.5 + self.epsilon)

        return weights


# Example usage:
adam = AdamOptimizer(learning_rate=0.069)
model = MLP()

out_func = func_with_model_state(train_step, model)

# xor data
data = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [[0], [1], [1], [0]]
# print(train_step(model,data,targets))

state = model.state_dict()


grad_func = value_and_grad(out_func)
iters = 500

# print(grad_func(state, data, targets))
model.load_state_dict(state)
for x, y in zip(data, targets):
    print(f"{y[0]} => {model(x)[0]:.2f}")
print("--")

for idx in range(iters):
    loss, grads = grad_func(state, data, targets)
    if idx % 50 == 0:
        print(f"loss: {loss}")
    state = adam.step(state, grads)

print("--")
model.load_state_dict(state)
for x, y in zip(data, targets):
    print(f"{y[0]} => {model(x)[0]:.2f}")
