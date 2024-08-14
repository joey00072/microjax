from __future__ import annotations

import numpy as np
import numbers

from typing import Any, Hashable, Iterable

PyTreeTypes = list | dict | tuple | Any


class PyNode:
    def __init__(
        self, node_type: type, metadata: Hashable = None, child_tree: "PyNode" = None
    ):
        self.node_type = node_type
        self.metadata = metadata
        self.child_tree = child_tree

    def __repr__(self):
        s = f"({self.node_type.__name__ if self.node_type !='leaf' else 'leaf'}"
        if isinstance(self.metadata, np.ndarray) or self.metadata:
            s += f":{self.metadata.__class__.__name__}"
        if self.child_tree is not None:
            s += f",{self.child_tree}"
        return s + ")"

    @staticmethod
    def from_iter(pytree) -> tuple[Hashable, "PyNode"]:
        raise NotImplementedError("Not implemented")

    @staticmethod
    def to_iter() -> PyTreeTypes:
        raise NotImplementedError("Not implemented")

    def __eq__(self, other: PyNode) -> bool:
        if self.node_type != other.node_type:
            return False
        if self.child_tree != other.child_tree:
            return False
        return True


class ListNode(PyNode):
    @staticmethod
    def to_iter(lst):
        return None, lst

    @staticmethod
    def from_iter(_, iterable):
        return list(iterable)


class DictNode(PyNode):
    @staticmethod
    def from_iter(keys, vals):
        return dict(zip(keys, vals))

    @staticmethod
    def to_iter(dct):
        keys, values = [], []
        for key, value in sorted(dct.items()):
            keys.append(key)
            values.append(value)
        return keys, values


class TupleNode(PyNode):
    @staticmethod
    def from_iter(_, tup):
        return tuple(tup)

    @staticmethod
    def to_iter(tup):
        return None, tup


node_types: dict[Hashable, PyNode | None] = {
    list: ListNode,
    dict: DictNode,
    tuple: TupleNode,
}


def tree_flatten(x: Any) -> tuple[list[Any], PyNode]:
    def _flatten(x: Any) -> tuple[Iterable, PyNode]:
        data_type = type(x)
        node_type = node_types.get(data_type)
        if node_type is None:
            return [x], PyNode(node_type="leaf", metadata=x, child_tree=None)

        node_metadata, children = node_type.to_iter(x)

        children_flat, child_trees = [], []
        for node in children:
            flat, tree = _flatten(node)
            children_flat.extend(flat)
            child_trees.append(tree)

        subtree = PyNode(
            data_type,  # store the base type instead of the specific node type
            node_metadata,
            tuple(child_trees),
        )
        return children_flat, subtree

    flatten, pytree = _flatten(x)
    return flatten, pytree


def tree_unflatten(flattened_list: list, tree: PyNode) -> Any:
    def _unflatten(flattened_list: list, tree: PyNode) -> Any:
        if tree.node_type == "leaf":
            return next(flattened_list)

        children = []
        for child_tree in tree.child_tree:
            children.append(_unflatten(flattened_list, child_tree))

        node_type = node_types[tree.node_type]
        return node_type.from_iter(tree.metadata, children)

    return _unflatten(iter(flattened_list[:]), tree)


def flatten_fun(func, in_tree):
    store = {}

    def flat_fun(*args_flat):
        pytree_args = tree_unflatten(args_flat, in_tree)
        out = func(*pytree_args)
        out_flat, out_tree = tree_flatten(out)
        assert len(store) == 0, "Store already has a value!"
        store["tree"] = out_tree
        return out_flat

    return flat_fun, store


def display_tree(node: PyNode, indent: str = "") -> None:
    if node.node_type == "leaf":
        print(f"{indent}Leaf: {node.metadata}")
    else:
        node_type_name = node.node_type.__name__ if node.node_type != "leaf" else "leaf"
        print(f"{indent}{node_type_name}: {node.metadata}")
        for child in node.child_tree:
            display_tree(child, indent + "    ")


# These functions create nested structures of ones or zeros that match the input structure


def nested_ones_like(item):
    """Create a nested structure of ones with the same shape as the input."""
    if isinstance(item, list):
        return [nested_ones_like(x) for x in item]
    if isinstance(item, tuple):
        return tuple(nested_ones_like(x) for x in item)
    if isinstance(item, dict):
        return {k: nested_ones_like(v) for k, v in item.items()}
    return 1.0 if isinstance(item, numbers.Number) else np.ones_like(item)


def nested_zero_like(item):
    """Create a nested structure of zeros with the same shape as the input."""
    if isinstance(item, list):
        return [nested_zero_like(x) for x in item]
    if isinstance(item, tuple):
        return tuple(nested_zero_like(x) for x in item)
    if isinstance(item, dict):
        return {k: nested_zero_like(v) for k, v in item.items()}
    return 0.0 if isinstance(item, numbers.Number) else np.zeros_like(item)


if __name__ == "__main__":
    x = [1, (2, {"a": 3, "b": 4}, 5), [6, 7]]
    flattened, tree = tree_flatten(x)
    print(x)
    print("\nTree structure:")
    display_tree(tree)
    print("\n")
    print("Flattened:", flattened)
    print("\n")

    reconstructed = tree_unflatten(flattened, tree)
    print("\nReconstructed:", reconstructed)
    assert x == reconstructed, "Reconstruction failed"
    print("Reconstruction successful!")
