import os
import click
import yaml
import geopandas as gpd
from shapely.geometry import box
from planet.api import ClientV1, auth, utils


@click.command()
@click.option('--config_path', '-c',
              default='cfgs/config_main.yaml',
              help='The configure yaml path.')
def main(config_path):
    # Load config yaml
    # config_path = 'cfgs/config_plt.yaml'
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Initial Planet client
    if auth.find_api_key() is None:
        utils.write_planet_json({'key': config['planet']['key']})
    client = ClientV1()

    # Load study area
    aoi = gpd.read_file(config['imagery']['geom'])[['geometry']]
    bbox = aoi.total_bounds

    # Start query
    # Download images and save out the tile grids.
    # Create dir
    if not os.path.exists(config['imagery']['quads_path']):
        os.mkdir(config['imagery']['quads_path'])

    # Create tile grids
    # Because the tiles are the same, so just use the first date to get tiles
    mosaic = client.get_mosaics(name_contains=config['imagery']['dates'][0]).get()['mosaics'][0]
    quads = client.get_quads(mosaic, bbox=bbox).items_iter(limit=10000)
    tiles = []
    ids = []
    for quad in quads:
        tiles.append(box(quad['bbox'][0], quad['bbox'][1],
                         quad['bbox'][2], quad['bbox'][3]))
        ids.append(quad['id'])
    tiles = gpd.GeoDataFrame({'tile': ids, 'geometry': tiles}, crs="EPSG:4326")
    # tiles = gpd.overlay(aoi, tiles)
    tiles = gpd.sjoin(left_df=tiles, right_df=aoi).drop(columns=['index_right'])
    file = os.path.join(config['imagery']['tile_path'], 'tiles_nicfi.geojson')
    tiles.to_file(file, driver='GeoJSON')

    # Download quads, and filter by the tiles
    for nm_contain in config['imagery']['dates']:
        mosaic = client.get_mosaics(name_contains=nm_contain).get()['mosaics'][0]
        quads = client.get_quads(mosaic, bbox=bbox).items_iter(limit=10000)
        for quad in quads:
            if quad['id'] in list(tiles['tile']):
                file = os.path.join(config['imagery']['quads_path'],
                                    mosaic['name'] + '_' + quad['id'] + '.tif')
                if not os.path.exists(file):
                    client.download_quad(quad).get_body().write(file)
                else:
                    print('File {} already exists.'.format(file))


if __name__ == '__main__':
    main()
