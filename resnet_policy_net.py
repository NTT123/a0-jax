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
        self, input_dims: int, num_actions: int, dim: int = 128, num_resblock: int = 4
    ):
        super().__init__()

        self.input_dims = input_dims
        self.num_actions = num_actions
        self.backbone = pax.Sequential(pax.Conv2D(1, dim, 1), pax.BatchNorm2D(dim))
        for _ in range(num_resblock):
            self.backbone >>= ResidualBlock(dim)
        self.action_head = pax.Sequential(
            ResidualBlock(dim),
            ResidualBlock(dim),
            pax.Conv2D(dim, num_actions, kernel_shape=input_dims, padding="VALID"),
        )
        self.value_head = pax.Sequential(
            ResidualBlock(dim),
            ResidualBlock(dim),
            pax.Conv2D(dim, 1, kernel_shape=input_dims, padding="VALID"),
        )

    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = jnp.reshape(x, [1, *self.input_dims] + [1])
        x = self.backbone(x)
        action_logits = self.action_head(x)
        value = jnp.tanh(self.value_head(x))
        return action_logits[0, 0, 0, :], value[0, 0, 0, 0]
