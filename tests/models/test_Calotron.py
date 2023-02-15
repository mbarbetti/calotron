import pytest
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer, RMSprop

chunk_size = int(1e4)

source = tf.random.normal(shape=(chunk_size, 16, 3))
target = tf.random.normal(shape=(chunk_size, 32, 9))


@pytest.fixture
def model():
    from calotron.models import Calotron, Discriminator, Transformer

    transf = Transformer(
        output_depth=target.shape[2],
        encoder_depth=8,
        decoder_depth=8,
        num_layers=2,
        num_heads=4,
        key_dim=None,
        encoder_pos_dim=None,
        decoder_pos_dim=None,
        encoder_pos_normalization=64,
        decoder_pos_normalization=64,
        encoder_max_length=source.shape[1],
        decoder_max_length=target.shape[1],
        ff_units=16,
        dropout_rate=0.1,
        pos_sensitive=False,
        residual_smoothing=True,
        output_activations="relu",
        start_token_initializer="zeros",
    )

    disc = Discriminator(
        latent_dim=8,
        output_units=1,
        output_activation="sigmoid",
        hidden_layers=2,
        hidden_units=32,
        dropout_rate=0.1,
    )

    calo = Calotron(transformer=transf, discriminator=disc)
    return calo


###########################################################################


def test_model_configuration(model):
    from calotron.models import Calotron, Discriminator, Transformer

    assert isinstance(model, Calotron)
    assert isinstance(model.transformer, Transformer)
    assert isinstance(model.discriminator, Discriminator)


def test_model_use(model):
    outputs = model((source, target))
    t_output, d_output_true, d_output_pred = outputs
    model.summary()

    test_t_shape = list(target.shape)
    test_t_shape[-1] = model.transformer.output_depth
    assert t_output.shape == tuple(test_t_shape)

    test_d_shape = [target.shape[0]]
    test_d_shape.append(model.discriminator.output_units)
    assert d_output_true.shape == tuple(test_d_shape)
    assert d_output_pred.shape == tuple(test_d_shape)


@pytest.mark.parametrize("metrics", [["bce"], None])
def test_model_compilation(model, metrics):
    from calotron.losses import CaloLoss

    loss = CaloLoss(alpha=0.1)
    t_opt = RMSprop(learning_rate=0.001)
    d_opt = RMSprop(learning_rate=0.001)
    model.compile(
        loss=loss,
        metrics=metrics,
        transformer_optimizer=t_opt,
        discriminator_optimizer=d_opt,
        transformer_upds_per_batch=1,
        discriminator_upds_per_batch=1,
    )
    assert isinstance(model.metrics, list)
    assert isinstance(model.transformer_optimizer, Optimizer)
    assert isinstance(model.discriminator_optimizer, Optimizer)
    assert isinstance(model.transformer_upds_per_batch, int)
    assert isinstance(model.discriminator_upds_per_batch, int)


def test_model_train(model):
    dataset = (
        tf.data.Dataset.from_tensor_slices((source, target))
        .batch(batch_size=512, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    from calotron.losses import CaloLoss

    loss = CaloLoss(alpha=0.1)
    t_opt = RMSprop(learning_rate=0.001)
    d_opt = RMSprop(learning_rate=0.001)
    model.compile(
        loss=loss,
        metrics=None,
        transformer_optimizer=t_opt,
        discriminator_optimizer=d_opt,
        transformer_upds_per_batch=1,
        discriminator_upds_per_batch=1,
    )
    model.fit(dataset, epochs=3)
