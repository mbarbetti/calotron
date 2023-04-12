import pytest
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer, RMSprop

CHUNK_SIZE = int(1e4)

source = tf.random.normal(shape=(CHUNK_SIZE, 8, 5))
target = tf.random.normal(shape=(CHUNK_SIZE, 4, 3))


@pytest.fixture
def model():
    from calotron.models import AuxClassifier, Calotron, Discriminator, Transformer

    transf = Transformer(
        output_depth=target.shape[2],
        encoder_depth=8,
        decoder_depth=8,
        num_layers=2,
        num_heads=4,
        key_dims=32,
        mlp_units=16,
        dropout_rates=0.1,
        seq_ord_latent_dims=16,
        seq_ord_max_lengths=[source.shape[1], target.shape[1]],
        seq_ord_normalizations=10_000,
        residual_smoothing=True,
        output_activations="relu",
        start_token_initializer="ones",
    )

    disc = Discriminator(
        latent_dim=8,
        output_units=1,
        output_activation="sigmoid",
        deepsets_num_layers=2,
        deepsets_hidden_units=32,
        dropout_rate=0.1,
    )

    aux = AuxClassifier(
        transformer=transf,
        output_depth=1,
        output_activation="sigmoid",
        dropout_rate=0.1,
    )

    calo = Calotron(transformer=transf, discriminator=disc, aux_classifier=aux)
    return calo


###########################################################################


def test_model_configuration(model):
    from calotron.models import AuxClassifier, Calotron, Discriminator, Transformer

    assert isinstance(model, Calotron)
    assert isinstance(model.transformer, Transformer)
    assert isinstance(model.discriminator, Discriminator)
    assert isinstance(model.aux_classifier, AuxClassifier)


def test_model_use(model):
    outputs = model((source, target))
    t_output, d_output_true, d_output_pred, a_output = outputs
    model.summary()

    test_t_shape = list(target.shape)
    test_t_shape[-1] = model.transformer.output_depth
    assert t_output.shape == tuple(test_t_shape)

    test_d_shape = [target.shape[0]]
    test_d_shape.append(model.discriminator.output_units)
    assert d_output_true.shape == tuple(test_d_shape)
    assert d_output_pred.shape == tuple(test_d_shape)

    test_a_shape = list(source.shape)
    test_a_shape[-1] = model.aux_classifier.output_depth
    assert a_output.shape == tuple(test_a_shape)


@pytest.mark.parametrize("metrics", [["bce"], None])
def test_model_compilation(model, metrics):
    from calotron.losses import PhotonClusterMatch

    loss = PhotonClusterMatch(
        lambda_adv=0.1,
        lambda_global=1.0,
        max_match_distance=0.005,
        adversarial_metric="binary-crossentropy",
    )
    t_opt = RMSprop(learning_rate=0.001)
    d_opt = RMSprop(learning_rate=0.001)
    a_opt = RMSprop(learning_rate=0.001)
    model.compile(
        loss=loss,
        metrics=metrics,
        transformer_optimizer=t_opt,
        discriminator_optimizer=d_opt,
        aux_classifier_optimizer=a_opt,
        transformer_upds_per_batch=1,
        discriminator_upds_per_batch=1,
        aux_classifier_upds_per_batch=1,
    )
    assert isinstance(model.metrics, list)
    assert isinstance(model.transformer_optimizer, Optimizer)
    assert isinstance(model.discriminator_optimizer, Optimizer)
    assert isinstance(model.aux_classifier_optimizer, Optimizer)
    assert isinstance(model.transformer_upds_per_batch, int)
    assert isinstance(model.discriminator_upds_per_batch, int)
    assert isinstance(model.aux_classifier_upds_per_batch, int)


@pytest.mark.parametrize(
    "adversarial_metrics", ["binary-crossentropy", "wasserstein-distance"]
)
def test_model_train(model, adversarial_metrics):
    dataset = (
        tf.data.Dataset.from_tensor_slices((source, target))
        .batch(batch_size=512, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    from calotron.losses import PhotonClusterMatch

    loss = PhotonClusterMatch(
        lambda_adv=0.1,
        lambda_global=1.0,
        max_match_distance=0.005,
        adversarial_metric=adversarial_metrics,
    )
    t_opt = RMSprop(learning_rate=0.001)
    d_opt = RMSprop(learning_rate=0.001)
    a_opt = RMSprop(learning_rate=0.001)
    model.compile(
        loss=loss,
        metrics=None,
        transformer_optimizer=t_opt,
        discriminator_optimizer=d_opt,
        aux_classifier_optimizer=a_opt,
        transformer_upds_per_batch=1,
        discriminator_upds_per_batch=1,
        aux_classifier_upds_per_batch=1,
    )
    model.fit(dataset, epochs=2)


def test_model_eval(model):
    from calotron.losses import PhotonClusterMatch

    loss = PhotonClusterMatch(
        lambda_adv=0.1,
        lambda_global=1.0,
        max_match_distance=0.005,
        adversarial_metric="binary-crossentropy",
    )
    t_opt = RMSprop(learning_rate=0.001)
    d_opt = RMSprop(learning_rate=0.001)
    a_opt = RMSprop(learning_rate=0.001)
    model.compile(
        loss=loss,
        metrics=None,
        transformer_optimizer=t_opt,
        discriminator_optimizer=d_opt,
        aux_classifier_optimizer=a_opt,
        transformer_upds_per_batch=1,
        discriminator_upds_per_batch=1,
        aux_classifier_upds_per_batch=1,
    )
    model.evaluate(source, target)
