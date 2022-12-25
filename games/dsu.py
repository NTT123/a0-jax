"""Disjoint set union in JAX.

Reference: https://cp-algorithms.com/data_structures/disjoint_set_union.html
"""

import jax
import jax.numpy as jnp
import pax

from utils import select_tree


class DSU(pax.Module):
    """Fast union of disjoint sets."""

    def __init__(self, N, get_all_roots_freq=None):
        super().__init__()
        self.parent = jnp.arange(0, N, 1, dtype=jnp.int32)
        self.size = jnp.ones((N,), dtype=jnp.int32)
        self.N = N
        self.get_all_roots_freq = get_all_roots_freq

    def masked_reset(self, mask):
        """Reset sets in mask"""
        parent = jnp.arange(0, self.N, 1, dtype=jnp.int32)
        size = jnp.ones((self.N,), dtype=jnp.int32)
        self.parent = jnp.where(mask, parent, self.parent)
        self.size = jnp.where(mask, size, self.size)

    def find_set_pure(self, element):
        """Find set but without updating parent.

        If `get_all_roots` method is called frequently,
        use python `for loop` for better performance on GPU.
        """

        if self.get_all_roots_freq is not None:
            for _ in range(self.get_all_roots_freq + 1):
                element = self.parent[element]
            return element

        def cond(u):
            return self.parent[u] != u

        def loop(u):
            return self.parent[u]

        return jax.lax.while_loop(cond, loop, element)

    def find_set(self, v):
        """Find the root of a set and update parent array."""
        root = self.find_set_pure(v)

        def cond(pu):
            _, u = pu
            return self.parent[u] != u

        def loop(pu):
            p, u = pu
            p = p.at[u].set(root)
            return p, self.parent[u]

        parent, _ = jax.lax.while_loop(cond, loop, (self.parent, v))
        self.parent = parent
        return root

    def get_all_roots(self):
        """Find roots of all elements.
        Update the parent array to make later calls faster.
        """
        v = jnp.arange(0, self.N, 1, jnp.int32)
        roots = jax.vmap(lambda s, v: s.find_set_pure(v), in_axes=(None, 0))(self, v)
        self.parent = roots
        return roots

    def union_sets(self, a, b):
        """Union two sets a and b"""
        a = self.find_set_pure(a)
        b = self.find_set_pure(b)

        def if_true(x, y, parent, size):
            x, y = select_tree(size[x] < size[y], (y, x), (x, y))
            parent = parent.at[y].set(x)
            size = size.at[x].add(size[y])
            return parent, size

        parent, size = if_true(a, b, self.parent, self.size)
        self.parent, self.size = select_tree(
            a != b, (parent, size), (self.parent, self.size)
        )

    def pp(self):
        """pretty print"""
        print("Parent:", self.parent, "Size:", self.size)
