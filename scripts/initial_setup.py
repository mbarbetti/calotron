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

export_dir = config_dir["export_dir"]
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

images_dir = config_dir["images_dir"]
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

report_dir = config_dir["report_dir"]
if not os.path.exists(report_dir):
    os.makedirs(report_dir)
