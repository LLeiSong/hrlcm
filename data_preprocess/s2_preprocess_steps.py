import logging
import click
import os
import re
import sys
import time
import yaml
from datetime import datetime
from os.path import join
from sentinelPot import ParserConfig, \
    parse_config, parse_catalog, \
    peps_maja_downloader, peps_maja_process


def _divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


@click.command()
@click.option('--config_path', '-c',
              default='cfgs/config_main.yaml',
              help='The configure yaml path.')
def main(config_path):
    # config_path = 'cfgs/config_main_s2.yaml'
    options = ParserConfig(config_path)
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    if not os.path.exists(join(config['dirs']['dst_dir'], options.log_dir)):
        os.mkdir(join(config['dirs']['dst_dir'], options.log_dir))

    # Set logging
    if options.log_dir is not None:
        log = "{}/{}/full_maja_process_{}.log" \
            .format(options.dst_dir,
                    options.log_dir,
                    datetime.now().strftime("%d%m%Y_%H%M"))
    else:
        log = "{}/full_maja_process_{}.log" \
            .format(options.dst_dir,
                    datetime.now().strftime("%d%m%Y_%H%M"))

    # Set up logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_format = "%(asctime)s::%(levelname)s::%(name)s::%(filename)s::%(lineno)d::%(message)s"
    logging.basicConfig(filename=log, filemode='w',
                        level=logging.INFO, format=log_format)
    logger = logging.getLogger(__name__)

    # Check destination path
    if options.dst_dir is None:
        logger.error("full_maja_process: must set a destination path for results.")
        sys.exit("full_maja_process: must set a destination path for results.")

    # ====================
    # read authentication file
    # ====================
    config = parse_config(options.auth)
    email = config['peps']['user']
    passwd = config['peps']['password']
    if email is None or passwd is None:
        print("Not valid email or passwd for peps.")
        logger.error("full_maja_process: Not valid email or passwd for peps.")
        sys.exit(-1)

    # Stage images to disk and get catalog
    # peps_downloader(options)
    prod, download_dict, storage_dict, size_dict = parse_catalog(options, logger)
    prod = list(set(download_dict.keys()))
    tiles_dup = list(map(lambda x: re.search("T[0-9]{2}[A-Z]{3}", x).group(0), prod))
    tiles = list(set(map(lambda x: re.search("T[0-9]{2}[A-Z]{3}", x).group(0), prod)))
    tiles.sort()
    tiles = list(_divide_chunks(tiles, 10))

    # Request maja
    # Create path for logs
    if not os.path.isdir(join(options.dst_dir, options.maja_log)):
        os.mkdir(join(options.dst_dir, options.maja_log))

    if options.start_date is not None:
        start_date = options.start_date
        if options.end_date is not None:
            end_date = options.end_date
        else:
            end_date = datetime.date.today().isoformat()
    for each in tiles:
        tiles_done = []
        wait_lens = []
        for tile in each:
            # Set logName for maja
            log_name = join(options.dst_dir, options.maja_log, '{}.log'.format(tile))
            if peps_maja_process(start_date, end_date, tile,
                                 log_name, email, passwd,
                                 logger=logger, no_download=options.no_download):
                tiles_done.append(tile)
                wait_lens.append(60 + 25 * (tiles_dup.count(tile) - 1))
                logger.info('full_maja_process: query maja for tile {} success.'.format(tile))
            else:
                logger.error('full_maja_process: query maja for tile {} fails.'.format(tile))
        time.sleep(max(wait_lens))

        # Download finished images
        for tile in tiles_done:
            log_name = join(options.dst_dir, options.maja_log, '{}.log'.format(tile))
            while True:
                if peps_maja_downloader(options.processed_dir, email, passwd, log_name, logger):
                    logger.info('full_maja_process: download imagery of tile {} success.'.format(tile))
                    break

    print('Request finish. Please check {} for details.'.format(log))


if __name__ == '__main__':
    main()
