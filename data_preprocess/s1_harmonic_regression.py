import click
import yaml
from sentinelPot import harmonic_fitting


@click.command()
@click.option('--config_path', '-c',
              default='cfgs/config_main.yaml',
              help='The configure yaml path.')
def main(config_path):
    # config_path = 'cfgs/config_main_sa_s1.yaml'
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    tile_index = '1205-978'
    for pol in ['VV', 'VH']:
        harmonic_fitting(str(tile_index), pol, config)


if __name__ == '__main__':
    main()
