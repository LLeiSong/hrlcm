import logging
import click
import sys
import yaml
import os
from os.path import join
from sentinelPot import ParserConfig, peps_maja_downloader


@click.command()
@click.option('--config_path', '-c',
              default='cfgs/config_main.yaml',
              help='The configure yaml path.')
def main(config_path):
    # config_path = 'cfgs/config_main_sa_s2.yaml'
    options = ParserConfig(config_path)

    # Set up logger
    log = join(options.dst_dir, options.log_dir, 's2_maja_download.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_format = "%(asctime)s::%(levelname)s::%(name)s::%(filename)s::%(lineno)d::%(message)s"
    logging.basicConfig(filename=log, filemode='w',
                        level=logging.INFO, format=log_format)
    logger = logging.getLogger(__name__)

    # Check destination path
    if options.dst_dir is None:
        sys.exit("full_maja_process: must set a destination path for results.")

    # ====================
    # read authentication file
    # ====================
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    email = config['peps']['user']
    passwd = config['peps']['password']
    if email is None or passwd is None:
        print("Not valid email or passwd for peps.")
        sys.exit(-1)

    log_names = os.listdir(join(options.dst_dir, options.maja_log))
    log_names = list(map(lambda fn: join(options.dst_dir, options.maja_log, fn), log_names))
    for log_name in log_names:
        print('Start to use {}'.format(log_name))
        while True:
            if peps_maja_downloader('/Volumes/elephant/sentinel2_level2',
                                    email, passwd, log_name, logger):
                break
    print('Request finish. Please check {} for details.'.format(log))


if __name__ == '__main__':
    main()
