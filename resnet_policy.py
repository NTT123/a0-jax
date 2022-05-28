import chex
import jax
import jax.numpy as jnp
import pax


class ResidualBlock(pax.Module):
    """A residual block of conv layers

    y(x) = x + F(x),

    where:
        F = BatchNorm >> relu >> Conv1 >> BatchNorm >> relu >> Conv2
    """

    def __init__(self, dim):
        super().__init__()
        self.batchnorm1 = pax.BatchNorm2D(dim, True, True)
        self.batchnorm2 = pax.BatchNorm2D(dim, True, True)
        self.conv1 = pax.Conv2D(dim, dim, 3)
        self.conv2 = pax.Conv2D(dim, dim, 3)

    def __call__(self, x):
        t = jax.nn.relu(self.batchnorm1(x))
        t = self.conv1(t)
        t = jax.nn.relu(self.batchnorm2(x))
        t = self.conv2(t)
        return x + t


class ResnetPolicyValueNet(pax.Module):
    """Residual Conv Policy-Value network.

    Two-head network:
                        ┌─> action head
    input ─> Backbone  ─┤
                        └─> value head
    """

    def __init__(
        self, input_dims, num_actions: int, dim: int = 128, num_resblock: int = 5
    ):
        super().__init__()

        self.input_dims = input_dims
        self.num_actions = num_actions
        self.backbone = pax.Sequential(pax.Conv2D(1, dim, 1), pax.BatchNorm2D(dim))
        for _ in range(num_resblock):
            self.backbone >>= ResidualBlock(dim)
        self.action_head = pax.Sequential(
            pax.Conv2D(dim, dim, 1),
            pax.BatchNorm2D(dim, True, True),
            jax.nn.relu,
            pax.Conv2D(dim, self.num_actions, kernel_shape=input_dims, padding="VALID"),
        )
        self.value_head = pax.Sequential(
            pax.Conv2D(dim, dim, 1),
            pax.BatchNorm2D(dim, True, True),
            jax.nn.relu,
            pax.Conv2D(dim, dim, kernel_shape=input_dims, padding="VALID"),
            pax.BatchNorm2D(dim, True, True),
            jax.nn.relu,
            pax.Conv2D(dim, 1, kernel_shape=1, padding="VALID"),
            pax.BatchNorm2D(1, True, True),
            jnp.tanh,
        )

    def __call__(self, x: chex.Array, batched: bool = False):
        """Compute the action logits and value.

        Support both batched and unbatched states.
        """
        x = x.astype(jnp.float32)
        if not batched:
            x = x[None]  # add the batch dimension
        x = x[..., None]  # add channel dimension
        x = self.backbone(x)
        action_logits = self.action_head(x)
        value = self.value_head(x)
        if batched:
            return action_logits[:, 0, 0, :], value[:, 0, 0, 0]
        else:
            return action_logits[0, 0, 0, :], value[0, 0, 0, 0]
