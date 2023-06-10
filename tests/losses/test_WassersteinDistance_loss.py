import pytest
import tensorflow as tf

from calotron.models.discriminators import Discriminator
from calotron.models.transformers import Transformer

CHUNK_SIZE = int(1e4)
source = tf.random.normal(shape=(CHUNK_SIZE, 8, 5))
target = tf.random.normal(shape=(CHUNK_SIZE, 4, 3))
weight = tf.random.uniform(shape=(CHUNK_SIZE, target.shape[1]))

transf = Transformer(
    output_depth=target.shape[2],
    encoder_depth=8,
    decoder_depth=8,
    num_layers=2,
    num_heads=4,
    key_dim=32,
    admin_res_scale="O(n)",
    mlp_units=128,
    dropout_rate=0.1,
    seq_ord_latent_dim=16,
    seq_ord_max_length=max(source.shape[1], target.shape[1]),
    seq_ord_normalization=10_000,
    enable_res_smoothing=True,
    output_activations="linear",
    start_token_initializer="ones",
)

disc = Discriminator(
    output_units=1,
    output_activation=None,
    latent_dim=8,
    deepsets_dense_num_layers=2,
    deepsets_dense_units=32,
    dropout_rate=0.1,
)


@pytest.fixture
def loss():
    from calotron.losses import WassersteinDistance

    loss_ = WassersteinDistance(
        warmup_energy=0.0,
        lipschitz_penalty=100.0,
        virtual_direction_upds=1,
        fixed_xi=10.0,
        sampled_xi_min=0.0,
        sampled_xi_max=1.0,
        epsilon=1e-12,
    )
    return loss_


###########################################################################


def test_loss_configuration(loss):
    from calotron.losses import WassersteinDistance

    assert isinstance(loss, WassersteinDistance)
    assert isinstance(loss.warmup_energy, float)
    assert isinstance(loss.lipschitz_penalty, float)
    assert isinstance(loss.virtual_direction_upds, int)
    assert isinstance(loss.fixed_xi, float)
    assert isinstance(loss.sampled_xi_min, float)
    assert isinstance(loss.sampled_xi_max, float)
    assert isinstance(loss.epsilon, float)
    assert isinstance(loss.name, str)


@pytest.mark.parametrize("sample_weight", [weight, None])
def test_loss_use(loss, sample_weight):
    out = loss.transformer_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=sample_weight,
        training=False,
    )
    assert out.numpy()
    out = loss.discriminator_loss(
        transformer=transf,
        discriminator=disc,
        source=source,
        target=target,
        sample_weight=sample_weight,
        training=False,
    )
    assert out.numpy()
