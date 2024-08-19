# even small version hope its helps (^^)

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable


class OPS:
    ADD = "add"
    MUL = "mul"


def add(*args):
    return bind_single(OPS.ADD, *args)


def mul(*args):
    return bind_single(OPS.MUL, *args)


class Interpreter:
    def __init__(self, level: int = 0, *args, **kwargs):
        self.level = level

    def process_primitive(self, prim, boxes, params):
        raise NotImplementedError


# =========================================================
# this is global stack that have level and interpreter type
STACK: list[Interpreter] = []


def push_interpreter(interpreter: Interpreter):
    STACK.append(interpreter)
    return STACK


def pop_interpreter() -> Interpreter:
    return STACK.pop()


@contextmanager
def interpreter_context(interpreter_type: Interpreter):
    stack_item = interpreter_type(level=len(STACK))
    push_interpreter(stack_item)
    try:
        yield stack_item
    finally:
        pop_interpreter()


# =========================================================


class Box:
    _interpreter: Interpreter

    def aval(self):
        raise NotImplementedError

    def full_lower(self):
        return self

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)


class EvalRules:
    def __init__(self):
        self.rules = {
            OPS.ADD: self.add,
            OPS.MUL: self.mul,
        }

    def __getitem__(self, op):
        return self.rules[op]

    def add(self, primals, *args):
        x, y = primals
        return [x + y]

    def mul(self, primals, *args):
        x, y = primals
        return [x * y]


class EvalInterpreter(Interpreter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rules = EvalRules()

    def process_primitive(self, prim, boxes, params):
        return self.rules[prim](boxes, *params)


def find_top_interpreter(args):
    interpreters = []
    for item in args:
        if isinstance(item, Box):
            interpreters.append(item._interpreter)

    if interpreters:
        return max(interpreters, key=lambda x: x.level)

    # if no interpreters are found, return the default EvalInterpreter
    return STACK[0]


def full_raise(interpreter: Interpreter | Any, out) -> Box | JVPBox:
    """
    if interpreter need values boxed
    if out is not boxed, box it (using interpreter.pure)
    ie. raise out to the box level
    """
    if not isinstance(out, Box):
        return interpreter.pure(out)
    return out


def bind(prim, *args, **params):
    interpreter = find_top_interpreter(args)
    # this will raise the boxes to the top level
    # eg converts primitive values to Boxes if interpreter is not the top level
    boxes = [full_raise(interpreter, arg) for arg in args]
    outs = interpreter.process_primitive(prim, boxes, params)
    return [out for out in outs]


def bind_single(prim, *args, **params):
    (out,) = bind(prim, *args, **params)
    return out


### Push EvalInterpreter at bottom of the stack
push_interpreter(EvalInterpreter())


# =========================================================
# Jacobian Vector Product (JVP)
# forward mode Automatic Differentiation
# =========================================================


class JVPBox(Box):
    def __init__(self, interpretor: Interpreter, primal, tangent) -> None:
        super().__init__()
        self._interpreter = interpretor
        self.primal = primal
        self.tangent = tangent

    def __repr__(self):
        return f"JVPBox (primal={self.primal}, tangent={self.tangent})"

    @property
    def aval(self):
        return self.primal.aval


class JVPRules:
    def __init__(self):
        self.rules = {
            OPS.ADD: self.add,
            OPS.MUL: self.mul,
        }

    # dont forget to return tuple(primals),tuple(tangents)
    def __getitem__(self, op):
        return self.rules[op]

    @staticmethod
    def add(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return (x + y,), (x_dot + y_dot,)

    @staticmethod
    def mul(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return (x * y,), (x_dot * y + x * y_dot,)


class JVPInterpreter(Interpreter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rules = JVPRules()

    def pure(self, val):
        return JVPBox(self, val, 0.0)

    def process_primitive(self, prim, boxes, params):
        primals = [box.primal for box in boxes]
        tangents = [box.tangent for box in boxes]
        op = self.rules[prim]
        primals_out, tangents_out = op(primals, tangents, **params)

        result = []
        for p, t in zip(primals_out, tangents_out):
            result.append(JVPBox(self, p, t))
        return result


def jvp_simple(func, primals, tangents):
    with interpreter_context(JVPInterpreter) as iptr:
        box_in = [JVPBox(iptr, x, t) for x, t in zip(primals, tangents)]
        out = func(*box_in)
        box_out = full_raise(iptr, out)
        primal_out, tangent_out = box_out.primal, box_out.tangent
    return primal_out, tangent_out


def deriv(function):
    def jvp_forward(input_value):
        _, gradient = jvp_simple(function, (input_value,), (1.0,))
        return gradient

    return jvp_forward


# =========================================================
if __name__ == "__main__":
    print("## Forward Mode Automatic Differentiation (JVP) ##")

    def func(x):
        return 3 * x * x * x + 2 * x * x + 2 * x

    x = 3.14

    f = func
    print(f"f(x) = {f(x)}")

    f = deriv(func)
    print(f"f'(x) = {f(x)}")

    f = deriv(deriv(func))
    print(f"f''(x) = {f(x)}")

    f = deriv(deriv(deriv(func)))
    print(f"f'''(x) = {f(x)}")

    print("-" * 100)


# =========================================================
# Vector Jacobian Product (VJP)
# reverse mode Automatic Differentiation
# =========================================================


class Node:
    def __init__(self, vjp: Callable, parents: list[Node]) -> None:
        self.vjp = vjp
        self.parents = parents

    @property
    def is_leaf(self):
        return len(self.parents) == 0


def get_leaf_nodes() -> Node:
    return Node(None, [])


class VJPRules:
    def __init__(self):
        self.rules = {
            OPS.ADD: self.add,
            OPS.MUL: self.mul,
        }
        """
        Jax define one of vjp or jvp rules
        it derives one from the other 
        but this is much more simple to understand
        """

    def __getitem__(self, op):
        return self.rules[op]

    def add(self, primals):
        x, y = primals

        def vjp_add(grad):
            return grad, grad

        return (x + y,), vjp_add

    def mul(self, primals):
        x, y = primals

        def vjp_mul(grad):
            return grad * y, grad * x

        return (x * y,), vjp_mul


class VJPInterpreter(Interpreter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rules = VJPRules()

    def pure(self, val):
        return VJPBox(self, val, get_leaf_nodes())

    def process_primitive(self, prim, boxes, params):
        primals_in = [box.primal for box in boxes]
        nodes_in = [box.node for box in boxes]
        op = self.rules[prim]
        primals_out, vjp_out = op(primals_in, **params)
        nodes_out = [Node(vjp_out, nodes_in)]
        result = []
        for p, n in zip(primals_out, nodes_out):
            result.append(VJPBox(self, p, n))
        return result


class VJPBox(Box):
    def __init__(self, interpreter: VJPInterpreter, primal, node: Node) -> None:
        super().__init__()
        self._interpreter = interpreter
        self.primal = primal
        self.node = node

    def __repr__(self):
        return f"VJPBox (primal={self.primal}, node={self.node})"

    def full_lower(self):
        return self

    def aval(self):
        return self.primal.aval


def vjp(f, *args):
    with interpreter_context(VJPInterpreter) as iptr:
        box_in = [VJPBox(iptr, x, get_leaf_nodes()) for x in args]
        out = f(*box_in)
        box_out = full_raise(iptr, out)
        in_nodes = [box.node for box in box_in]
        out_node = box_out.node
        primal_out = box_out.primal

        def func_vjp(grad):
            return backward_pass(in_nodes, out_node, grad)

    return primal_out, func_vjp


def backward_pass(in_nodes, out_node, gradient):
    node_map = {id(out_node): gradient}

    topo_sorted = toposort(out_node)
    for node in topo_sorted:
        node_grad = node_map.pop(id(node))

        input_grads = node.vjp(node_grad)

        for input_grad, parent in zip(input_grads, node.parents):
            parent_id = id(parent)
            node_map[parent_id] = add_grads(node_map.get(parent_id), input_grad)

    return [node_map.get(id(node)) for node in in_nodes]


def add_grads(grad1, grad2):
    if grad1 is None:
        return grad2
    return grad1 + grad2


def toposort(end_node):
    def _toposort(seen, node):
        result = []
        if id(node) not in seen:
            seen.add(id(node))
            for p in node.parents:
                result.extend(_toposort(seen, p))
            result.append(node)
        return result

    return reversed([n for n in _toposort(set(), end_node) if n.parents])


def grad(func):
    def grad_func(*args):
        _, backward = vjp(func, *args)
        return backward(1)[0]

    return grad_func


def func(x):
    # return x*x
    return 3 * x * x * x + 2 * x * x + 2 * x


if __name__ == "__main__":
    x = 3.14

    print("## Reverse Mode Automatic Differentiation (VJP) ##\n")
    f = func
    print(f"f(x) = {f(x)}")

    f = grad(func)
    print(f"f'(x) = {f(x)}")

    f = grad(grad(func))
    print(f"f''(x) = {f(x)}")

    f = grad(grad(grad(func)))
    print(f"f'''(x) = {f(x)}")

    print("-" * 100, "\n")

    print("Composition of Forward and Backward\n")
    print(f"Forward on Backward {grad(deriv(func))(x)}")
    print(f"Backward on Forward {deriv(grad(func))(x)}")
