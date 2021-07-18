import geopandas as gpd


# Open KML support
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

# Filter out tiles
tiles = gpd.read_file('data/geoms/hls_s2_tiles.kml')
bry = gpd.read_file('data/geoms/tiles_nicfi.geojson').dissolve()
tiles = gpd.overlay(tiles, bry, how='intersection')

# Save out tiles as text
tiles = tiles['Name']
tiles.to_csv('data/HLS/tiles.txt',
             header=False, sep='\t', index=False)
