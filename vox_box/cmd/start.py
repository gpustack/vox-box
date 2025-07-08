import argparse
import asyncio
import logging
import os

from vox_box.logging import setup_logging
from vox_box.config import Config
from vox_box.server.model import ModelInstance
from vox_box.server.server import Server
from vox_box.utils.model import parse_and_set_cuda_visible_devices


logger = logging.getLogger(__name__)


class OptionalBoolAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(OptionalBoolAction, self).__init__(
            option_strings, dest, nargs=0, **kwargs
        )
        self.default = None

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)


def setup_start_cmd(subparsers: argparse._SubParsersAction):
    parser_server: argparse.ArgumentParser = subparsers.add_parser(
        "start",
        help="Run vox-box api server.",
        description="Run vox-box api server.",
    )

    group = parser_server.add_argument_group("Common settings")
    group.add_argument(
        "-d",
        "--debug",
        action=OptionalBoolAction,
        help="Enable debug mode.",
        default=False,
    )

    group = parser_server.add_argument_group("Server settings")
    group.add_argument(
        "--host",
        type=str,
        help="Host to bind the server to.",
        default="0.0.0.0",
    )
    group.add_argument(
        "--port",
        type=int,
        help="Port to bind the server to.",
        default=80,
    )
    group.add_argument(
        "--model",
        type=str,
        help="Main model path.",
    )
    group.add_argument(
        "--device",
        type=str,
        help="Binding device, cuda:0.",
        default="cpu",
    )
    group.add_argument(
        "--huggingface-repo-id",
        type=str,
        help="Huggingface repo id for the estimate model.",
    )

    group.add_argument(
        "--model-scope-model-id",
        type=str,
        help="Model scope model id for the estimate model.",
    )

    group.add_argument(
        "--data-dir",
        type=str,
        help="Directory to store download model data. Default is OS specific.",
    )

    logger.info("Setting up start command.")
    parser_server.set_defaults(func=run)


def run(args: argparse.Namespace):
    try:
        cfg = parse_args(args)
        setup_logging(cfg.debug)
        parse_and_set_cuda_visible_devices(cfg)

        logger.info("Starting with arguments: %s", args._get_kwargs())

        run_model_instance(cfg)
        run_server(cfg)
    except Exception as e:
        logger.fatal(e)


def run_model_instance(cfg: Config):
    model_instance = ModelInstance(cfg)
    model_instance.run()


def run_server(cfg: Config):
    server = Server(config=cfg)
    asyncio.run(server.start())


def parse_args(args: argparse.Namespace) -> Config:
    validate_args(args)

    cfg = Config()
    cfg.debug = args.debug
    cfg.host = args.host
    cfg.port = args.port
    cfg.device = args.device
    cfg.model = args.model
    cfg.huggingface_repo_id = args.huggingface_repo_id
    cfg.model_scope_model_id = args.model_scope_model_id
    cfg.data_dir = args.data_dir or get_data_dir()
    cfg.cache_dir = os.path.join(cfg.data_dir, "cache")

    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    return cfg


def validate_args(args: argparse.Namespace):
    if (
        args.model is None
        and args.huggingface_repo_id is None
        and args.model_scope_model_id is None
    ):
        raise Exception(
            "One of model, huggingface-repo-id or model-scope-model-id is required."
        )


def get_data_dir():
    app_name = "vox-box"
    if os.name == "nt":  # Windows
        data_dir = os.path.join(os.environ["APPDATA"], app_name)
    elif os.name == "posix":
        data_dir = f"/var/lib/{app_name}"
    else:
        raise Exception("Unsupported OS")

    return os.path.abspath(data_dir)
