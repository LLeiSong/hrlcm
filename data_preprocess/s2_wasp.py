from sentinelPot import s2_wasp_batch
from zipfile import ZipFile
import yaml
import os
import click
from os.path import join


@click.command()
@click.option('--config_path', '-c',
              default='cfgs/config_main.yaml',
              help='The configure yaml path.')
def main(config_path):
    # # Read yaml
    # # config_path = 'cfgs/config_main_sa_s2.yaml'
    # with open(config_path, 'r') as yaml_file:
    #     config = yaml.safe_load(yaml_file)

    # # Unzip files
    # download_path = join(config['dirs']['dst_dir'], config['dirs']['processed_path'])
    # fnames = os.listdir(download_path)
    # fnames = list(filter(lambda fn: '.zip' in fn, fnames))
    # for fname in fnames:
    #     ZipFile(join(download_path, fname)).extractall(download_path)

    # Do WASP
    # Make sure run docker before
    s2_wasp_batch(config_path)


if __name__ == '__main__':
    main()
