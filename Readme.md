# MicroJAX

<p align="center">
  <img src="grad_plot.png" alt="Plot of tanh(x) and its First Four Derivatives" width="600">
</p>

`python tanh.py`

Microjax is function transformation engine like JAX or MLX,<br>
it’s got forward mode and reverse mode automatic differentiation support!, and support for nested structures (PyTrees). 🌳

## 🗂️ What's Inside?

- **`microjax.py`**: The magic of auto-diff happens here. 
- **`pytree.py`**:  🌳 Flatten and unflatten those nested data structures
- **`nn.py`**: Build a simple neural net and watch it learn XOR! 🧠
- **`tanh.py`**: Visualize the `tanh` function and its first four derivatives. 📈
- **`picojax.py`**: A extra small version of microjax.py (only +,* ops)

## 🚀 Quick Start
```python
from microjax import grad

def f(x):
    return x*x+2*x+3
print(f"f(3.0) = {f(3.0)}")
print(f"f'(3.0) = {grad(f)(3.0)}")
```

```bash
❯ python dev.py
f(3.0) = 18.0
f'(3.0) = 8.0

```

## Neural Net 
```zsh
❯ python nn.py 

0 => 0.50
1 => 0.34
1 => 0.44
0 => 0.32
--
loss: 0.2760428769255213
loss: 0.004209124188658117
loss: 0.000980696758933267
loss: 0.0005531283006194049
loss: 0.0003506475890801604
loss: 0.00023928890318040665
loss: 0.00017250868939842852
loss: 0.00012976140589010524
loss: 0.00010094563548150575
loss: 8.068691802714326e-05
--
0 => 0.00
1 => 0.99
1 => 0.99
0 => 0.01
```

<br>

Look into microjax.py 
```bash
❯ python microjax.py
## Forward Mode Automatic Differentiation (JVP) ##
f(x) = 118.87663200000001
f'(x) = 103.2964
f''(x) = 60.519999999999996
f'''(x) = 18.0
------------------------------------------------------------------------------

## Reverse Mode Automatic Differentiation (VJP) ##
f(x) = 118.87663200000001
f'(x) = 103.2964
f''(x) = 60.519999999999996
f'''(x) = 18
------------------------------------------------------------------------------ 

## Composition of Forward and Backward ##
Forward on Backward 60.519999999999996
Backward on Forward 60.519999999999996

------------------------------------------------------------------------------

## pytree.py ##
{'hi': 0.01493120808257803, 'there': [1.0, 0.0]}

------------------------------------------------------------------------------
MicroJAX:  0.01493120808257803
JAX:  0.014931838
```
## Limitations
- Only supports scalars
- slicing broadcating is NOT supported, but you can use numpy instead scalers
    - Adding ndarray support add complexity to the codebase, Keeping it Micro

## 📜 License

MIT License.

---
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/R6R8KQTZ5)