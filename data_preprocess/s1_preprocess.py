import click
from sentinelPot import s1_preprocess


@click.command()
@click.option('--config_path', '-c',
              default='cfgs/config_main.yaml',
              help='The configure yaml path.')
def main(config_path):
    # config_path = 'cfgs/config_main_sa_s1.yaml'
    s1_preprocess(config_path, query=False)


if __name__ == '__main__':
    main()

