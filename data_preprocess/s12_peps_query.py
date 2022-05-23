import click
from sentinelPot import peps_downloader, ParserConfig


@click.command()
@click.option('--config_path', '-c',
              default='cfgs/config_main.yaml',
              help='The configure yaml path.')
def main(config_path):
    # config_path = 'cfgs/config_main_sa_s1.yaml'
    options = ParserConfig(config_path)
    peps_downloader(options)


if __name__ == '__main__':
    main()

