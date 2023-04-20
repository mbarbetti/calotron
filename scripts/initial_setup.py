import os

import yaml

# +--------------------------+
# |   Directories creation   |
# +--------------------------+

with open("config/directories.yml") as file:
    config_dir = yaml.full_load(file)

data_dir = config_dir["data_dir"]
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

models_dir = config_dir["models_dir"]
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

images_dir = config_dir["images_dir"]
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

reports_dir = config_dir["reports_dir"]
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)
