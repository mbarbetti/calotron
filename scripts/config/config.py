import os
import yaml
from argparse import ArgumentParser


here = os.path.dirname(__file__)
parser = ArgumentParser(description="scripts configuration")

parser.add_argument("--interactive", action="store_true")
parser.add_argument("--no-interactive", dest="interactive", action="store_false")
parser.set_defaults(interactive=True)

parser.add_argument("-D", "--data_dir", default="./data")
parser.add_argument("-E", "--export_dir", default="./models")
parser.add_argument("-I", "--images_dir", default="./images")
parser.add_argument("-R", "--report_dir", default="./html")
config_dir = dict()

parser.add_argument("-s", "--server", default="https://hopaas.cloud.infn.it")
parser.add_argument("-t", "--token", default="user-api-token")
config_hopaas = dict()

args = parser.parse_args()

if args.interactive:
    data_dir = input(f"Path for the data directory (default: '{args.data_dir}'): ")
    config_dir["data_dir"] = data_dir if not (data_dir == "") else args.data_dir

    export_dir = input(f"Path for the export directory (default: '{args.export_dir}'): ")
    config_dir["export_dir"] = export_dir if not (export_dir == "") else args.export_dir

    images_dir = input(f"Path for the images directory (default: '{args.images_dir}'): ")
    config_dir["images_dir"] = images_dir if not (images_dir == "") else args.images_dir

    report_dir = input(f"Path for the report directory (default: '{args.report_dir}'): ")
    config_dir["report_dir"] = report_dir if not (report_dir == "") else args.report_dir

    server = input(f"Address of the Hopaas service (default: '{args.server}'): ")
    config_hopaas["server"] = server if not (server == "") else args.server

    token = input(f"API token to access the Hopaas service (default: '{args.token}'): ")
    config_hopaas["token"] = token if not (token == "") else args.token
else:
    config_dir["data_dir"] = args.data_dir
    config_dir["export_dir"] = args.export_dir
    config_dir["images_dir"] = args.images_dir
    config_dir["report_dir"] = args.report_dir

    config_hopaas["server"] = args.server
    config_hopaas["token"] = args.token

with open(f"{here}/directories.yml", "w") as file:
    yaml.dump(config_dir, file)

with open(f"{here}/hopaas.yml", "w") as file:
    yaml.dump(config_hopaas, file)
