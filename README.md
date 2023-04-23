<div align="center">
  <img alt="calotron logo" src="https://raw.githubusercontent.com/mbarbetti/calotron/main/.github/images/calotron-logo.png" width="600"/>
</div>

<h2 align="center">
  <em>Transformer-based models to fast-simulate the LHCb ECAL detector</em>
</h2>

<p align="center">
  <a href="https://www.tensorflow.org/versions"><img alt="TensorFlow versions" src="https://img.shields.io/badge/tensorflow-2.10–2.12-f57000?style=flat"></a>
  <a href="https://www.python.org/downloads"><img alt="Python versions" src="https://img.shields.io/badge/python-3.7–3.11-blue?style=flat"></a>
  <a href="https://pypi.python.org/pypi/calotron"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/calotron"></a>
  <a href="https://github.com/mbarbetti/calotron/blob/main/LICENSE"><img alt="GitHub - License" src="https://img.shields.io/github/license/mbarbetti/calotron"></a>
</p>

<p align="center">
  <a href="https://github.com/mbarbetti/calotron/actions/workflows/tests.yml"><img alt="GitHub - Tests" src="https://github.com/mbarbetti/calotron/actions/workflows/tests.yml/badge.svg?branch=main"></a>
  <a href="https://codecov.io/gh/mbarbetti/calotron"><img alt="Codecov" src="https://codecov.io/gh/mbarbetti/calotron/branch/main/graph/badge.svg?token=DRG8BWC9RR"></a>
</p>

<p align="center">
  <a href="https://github.com/mbarbetti/calotron/actions/workflows/style.yml"><img alt="GitHub - Style" src="https://github.com/mbarbetti/calotron/actions/workflows/style.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

<!--
[![Docker - Version](https://img.shields.io/docker/v/mbarbetti/calotron?label=docker)](https://hub.docker.com/r/mbarbetti/calotron)
-->

### Transformer
The Transformer architecture is freely inspired by Vaswani et al. [[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)] and Dosovitskiy et al. [[arXiv:2010.11929](https://arxiv.org/abs/2010.11929)].

<div align="center">
  <img alt="calotron transformer architecture" src="https://raw.githubusercontent.com/mbarbetti/calotron/main/.github/images/transf-scheme.png" width="400"/>
</div>

### Discriminator
The Discriminator is implemented through the Deep Sets model proposed by Zaheer et al. [[arXiv:1703.06114](https://arxiv.org/abs/1703.06114)] and its architecture is freely inspired by what developed by the ATLAS Collaboration for flavor tagging [[ATL-PHYS-PUB-2020-014](https://cds.cern.ch/record/2718948)].

<div align="center">
  <img alt="calotron discriminator architecture" src="https://raw.githubusercontent.com/mbarbetti/calotron/main/.github/images/disc-scheme.png" width="800"/>
</div>

### Credits
Transformer implementation freely inspired by the TensorFlow tutorial [Neural machine translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer).
