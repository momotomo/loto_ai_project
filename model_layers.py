"""Custom Keras layers for loto_ai model variants.

Import this module before calling tf.keras.models.load_model on any model
that uses custom layers (e.g. settransformer variant).  The
@register_keras_serializable decorator registers each class into Keras'
global serialization registry so that load_model can reconstruct the
objects without an explicit custom_objects mapping.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="loto_ai")
class SetAttentionBlock(tf.keras.layers.Layer):
    """Multi-head self-attention over the *set* dimension of a 4-D tensor.

    Applies a single Scaled Dot-Product Attention block (SAB) over the
    set-element axis of a batch of draw-sequences, enabling elements within
    the same draw to interact before the subsequent mean-pooling step.

    Input shape : ``(batch, lookback, set_cardinality, features)``
    Output shape: ``(batch, lookback, set_cardinality, features)``

    The block flattens the batch and lookback axes so that
    ``MultiHeadAttention`` sees a sequence of length ``set_cardinality``,
    applies residual + LayerNorm, then restores the original rank.

    Args:
        num_heads: Number of attention heads.
        key_dim:   Dimension of each attention head key/query projection.
        **kwargs:  Forwarded to ``tf.keras.layers.Layer``.
    """

    def __init__(self, num_heads: int, key_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)

    def build(self, input_shape):
        # Cache static spatial dimensions for use in call()
        self._lookback = int(input_shape[1])
        self._set_cardinality = int(input_shape[2])
        self._features = int(input_shape[3])

        self._mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            name="sab_mha",
        )
        self._norm = tf.keras.layers.LayerNormalization(name="sab_norm")
        # Force weight creation by calling build on sub-layers.
        # MultiHeadAttention.build() expects (query_shape, value_shape).
        qkv_shape = [None, self._set_cardinality, self._features]
        self._mha.build(qkv_shape, qkv_shape)
        self._norm.build([None, self._set_cardinality, self._features])
        super().build(input_shape)

    def call(self, x, training=None):  # noqa: D102
        # x: (batch, lookback, set_cardinality, features)
        batch = tf.shape(x)[0]

        # Merge batch and lookback → (batch*lookback, set_cardinality, features)
        x_flat = tf.reshape(
            x,
            [batch * self._lookback, self._set_cardinality, self._features],
        )

        # Self-attention: every element attends to every other in the same draw
        attn_out = self._mha(x_flat, x_flat, training=training)

        # Residual connection + layer normalisation
        x_flat = self._norm(x_flat + attn_out, training=training)

        # Restore original shape: (batch, lookback, set_cardinality, features)
        return tf.reshape(
            x_flat,
            [-1, self._lookback, self._set_cardinality, self._features],
        )

    def get_config(self):  # noqa: D102
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
        return config
