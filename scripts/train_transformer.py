import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from calotron.models import Transformer
from calotron.callbacks.schedulers import AttentionDecay
from calotron.simulators import Simulator


ORDER = True
POSITION = True
BATCHSIZE = 128
EPOCHS = 25
DTYPE = tf.float32


npzfile = np.load("data/train-data-demo.npz")
photon = npzfile["photon"][:, ::-1, :]
cluster = npzfile["cluster"]
cluster_labels = npzfile["cluster_labels"]

#print(f"photon {photon.shape}\n", photon)
#print(f"cluster {cluster.shape}\n", cluster)
#print(f"cluster labels {cluster_labels.shape}\n", cluster_labels)

chunk_size = photon.shape[0]

photon_shuffled = np.zeros_like(photon)
cluster_shuffled = np.zeros_like(cluster)
cluster_labels_shuffled = np.zeros_like(cluster_labels)
for i in range(chunk_size):
  photon_order = np.random.permutation(photon.shape[1])
  photon_shuffled[i] = photon[i, photon_order, :]
  cluster_order = np.random.permutation(cluster.shape[1])
  cluster_shuffled[i] = cluster[i, cluster_order, :]
  cluster_labels_shuffled[i] = cluster_labels[i, cluster_order, :]

#print(f"photon SHUFFLED {photon_shuffled.shape}\n", photon_shuffled)
#print(f"cluster SHUFFLED {cluster_shuffled.shape}\n", cluster_shuffled)
#print(f"cluster labels SHUFFLED {cluster_labels_shuffled.shape}\n", cluster_labels_shuffled)

if ORDER:
  photon_final = photon
  cluster_final = cluster
  cluster_labels_final = cluster_labels
else:
  photon_final = photon_shuffled
  cluster_final = cluster_shuffled
  cluster_labels_final = cluster_labels_shuffled

with tf.device("/gpu:0"):
  X = tf.cast(photon_final, dtype=DTYPE)
  Y = tf.cast(cluster_final, dtype=DTYPE)
  Y_labels = tf.cast(cluster_labels_final, dtype=DTYPE)

train_ds = (tf.data.Dataset.from_tensor_slices(((X, Y), Y_labels))
            .shuffle(chunk_size)
            .batch(BATCHSIZE, drop_remainder=True)
            .cache()
            .prefetch(tf.data.AUTOTUNE))

model = Transformer(output_depth=Y.shape[2],
                    encoder_depth=32,
                    decoder_depth=32,
                    num_layers=5,
                    num_heads=4,
                    key_dim=64,
                    encoder_pos_dim=16,
                    decoder_pos_dim=16,
                    encoder_pos_normalization=128,
                    decoder_pos_normalization=128,
                    encoder_max_length=X.shape[1],
                    decoder_max_length=Y.shape[1],
                    ff_units=256,
                    dropout_rate=0.1,
                    pos_sensitive=POSITION,
                    residual_smoothing=True,
                    output_activations=["linear", "linear", "sigmoid"],
                    dtype=DTYPE)

output = model((X, Y))
model.summary()

model.compile(loss="mse", optimizer="adam", metrics=["mae"])
sched = AttentionDecay(model.optimizer, d_model=64, warmup_steps=2000)

history = model.fit(train_ds, epochs=EPOCHS, callbacks=[sched])

sim = Simulator(model, start_token=Y[:, 0, :])
out = sim(X, max_length=Y.shape[1])

timestamp = str(datetime.now()).split(".")[0]
timestamp = timestamp.replace("-", "")
timestamp = timestamp.replace(" ", "-")
version = ""
for time, unit in zip(timestamp.split(":"), ["h", "m", "s"]):
  version += time + unit   # YYYYMMDD-HHhMMmSSs

tag = ""
if ORDER:
  tag += "ord"
else:
  tag += "dis"
tag += "+"
if POSITION:
  tag += "pos"
else:
  tag += "nopos"

plt.figure(dpi=100)
plt.hist(Y_labels[:,:,0].numpy().flatten(), bins=100, label="Training data")
plt.hist(out[:,:,0].numpy().flatten(), bins=100, label="Transformer output", histtype='step', linewidth=2)
plt.yscale('log')
plt.legend()
plt.xlabel("$x$ coordinate")
plt.savefig(f"img/{version}-{tag}-x.png")

plt.figure(dpi=100)
plt.hist(Y_labels[:,:,1].numpy().flatten(), bins=100, label="Training data")
plt.hist(out[:,:,1].numpy().flatten(), bins=100, label="Transformer output", histtype='step', linewidth=2)
plt.yscale('log')
plt.legend()
plt.xlabel("$y$ coordinate")
plt.savefig(f"img/{version}-{tag}-y.png")

plt.figure(dpi=100)
plt.hist(Y_labels[:,:,2].numpy().flatten(), bins=100, label="Training data")
plt.hist(out[:,:,2].numpy().flatten(), bins=100, label="Transformer output", histtype='step', linewidth=2)
plt.yscale('log')
plt.legend()
plt.xlabel("Preprocessed energy [a.u]")
plt.savefig(f"img/{version}-{tag}-energy.png")

plt.figure(figsize=(20, 10), dpi=80)
plt.subplot(1,2,1)
plt.imshow(Y_labels[:64, :, 2].numpy(), aspect='auto', interpolation='none')
plt.subplot(1,2,2)
plt.imshow(out[:64, :, 2].numpy(), aspect='auto', interpolation='none')
plt.savefig(f"img/{version}-{tag}-energy-map.png")

print("All figures exported correctly!")
plt.close()
