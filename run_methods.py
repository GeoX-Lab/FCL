import json
import argparse
import os.path

from trainer import train


def main():
    exps = ['finetune', 'ewc', 'lwf', 'bic', 'wa', 'coil']
    base_dir = './exps'

    for md in exps:
        print(f'*****************************=======================running method{md}======================********************************')

        method_dir = os.path.join(base_dir, md+'.json')

        args = setup_parser(method_dir).parse_args()
        param = load_json(args.config)
        args = vars(args)  # Converting argparse Namespace to a dict.
        args.update(param)  # Add parameters from json

        train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser(method):
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default=method,
                        help='Json file of settings.')

    parser.add_argument("--data_dir_folder", type=str, default=None,
                        help="fre combination data folders")

    return parser


if __name__ == '__main__':
    main()
