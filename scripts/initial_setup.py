import os
import yaml


# +--------------------------+
# |   Directories creation   |
# +--------------------------+

with open("config/directories.yml") as file:
  config = yaml.full_load(file)

data_dir = config["data_dir"]
if not os.path.exists (data_dir):
  os.makedirs(data_dir)

export_dir = config["export_dir"]
if not os.path.exists (export_dir):
  os.makedirs(export_dir)

images_dir = config["images_dir"]
if not os.path.exists (images_dir):
  os.makedirs(images_dir)

report_dir = config["report_dir"]
if not os.path.exists (report_dir):
  os.makedirs(report_dir)
