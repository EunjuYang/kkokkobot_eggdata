import argparse
import yaml

def load_configuration(path):
    with open(path, 'r') as f_cfg:
        config = yaml.load(f_cfg, Loader=yaml.SafeLoader)
    return config

class parser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='KKoKKo-BOT Training')

        self._add_conf_path()
        self._add_cuda_setting()
        self._add_seed_to_parser()
        self._add_mlflow_server()

    def _add_conf_path(self):
        self.parser.add_argument(
            '--conf-path',
            dest='conf_path',
            default='./conf/resnet18.yml',
            type=str,
            help='configuration file path'
        )

    def _add_cuda_setting(self):
        self.parser.add_argument(
            '--cuda',
            dest='cuda',
            default=True,
            type=bool
        )

    def _add_seed_to_parser(self):
        self.parser.add_argument(
            '--seed',
            dest='seed',
            default=1,
            type=int,
            help='set random seed for reproduction'
        )

    def _add_mlflow_server(self):
        self.parser.add_argument(
            '--mlflow',
            dest='mlflow',
            default='127.0.0.1:5000',
            type=str
        )
