from argparse import ArgumentParser

ADV_METRICS = ["bce", "wass"]


def argparser_preprocessing(description=None) -> ArgumentParser:
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-F",
        "--filename",
        default="./data/LamarrCaloTraining.root",
        help="path of the files from which downoloading data (default: './data/LamarrCaloTraining.root')",
    )
    parser.add_argument(
        "-M",
        "--max_files",
        default=10,
        help="maximum number of files from which downloading data (default: 10)",
    )
    parser.add_argument(
        "-C",
        "--chunk_size",
        default=-1,
        help="maximum number of instancens downloaded from the overall files (default:-1)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="arrange a few features and a limited number of particles per event (default: False)",
    )
    parser.add_argument("--no-demo", dest="weights", action="store_false")
    parser.set_defaults(demo=False)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print various pandas' DataFrames once created (default: False)",
    )
    parser.add_argument("--no-verbose", dest="saving", action="store_false")
    parser.set_defaults(verbose=False)
    return parser


def argparser_training(model, adv_learning=True, description=None) -> ArgumentParser:
    parser = ArgumentParser(description=description)
    if adv_learning:
        parser.add_argument(
            "-a",
            "--adv_metric",
            required=True,
            choices=ADV_METRICS,
            help="adversarial metric to be used for training",
        )
    parser.add_argument(
        "-C",
        "--chunk_size",
        default=-1,
        help="maximum number of instancens to be used for training/validation (default: -1)",
    )
    parser.add_argument(
        "-T",
        "--train_ratio",
        default=0.7,
        help="fraction of instances to be used for training (default: 0.7)",
    )
    parser.set_defaults(fullsim=True)
    parser.add_argument(
        "--weights",
        action="store_true",
        help=f"train the {model} model using weights when available (default: True)",
    )
    parser.add_argument("--no-weights", dest="weights", action="store_false")
    parser.set_defaults(weights=True)
    parser.add_argument(
        "--test",
        action="store_true",
        help="enable overwriting for model, images and reports since test execution (default: False)",
    )
    parser.add_argument("--no-test", dest="test", action="store_false")
    parser.set_defaults(test=False)
    parser.add_argument(
        "--saving",
        action="store_true",
        help="enable to save the trained model and all the images produced (default: False)",
    )
    parser.add_argument("--no-saving", dest="saving", action="store_false")
    parser.set_defaults(saving=False)
    return parser
