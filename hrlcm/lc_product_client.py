import yaml


class lc_ref:
    """
    A class to conveniently parse land cover products.
    Include:
    FROM-GLC10 (http://data.ess.tsinghua.edu.cn)
    CCI LAND COVER S2 prototype land cover map of Africa 2016 (http://2016africalandcover20m.esrin.esa.int)
    GFSAD cropland extent (https://lpdaac.usgs.gov/products/gfsad30afcev001/)
    Copernicus global land cover map (https://lcviewer.vito.be/2015)
    """
    def __init__(self, config):
        """
        :param config: the configure dictionary
        :type config: dict
        """
        if config['lc_ref']['year_cglc'] is not None:
            year_cglc = config['lc_ref']['year_cglc']
        else:
            print('No year for CGLC set, use the most recent year.')
            year_cglc = 2018
        years = {'FROM': 2017, 'CCI': 2016, 'GFSAD': 2015, 'CGLC': year_cglc}
        base_links = {'FROM': 'http://data.ess.tsinghua.edu.cn/data',
                      'CCI': 'some_personal_link',
                      'GFSAD': 'https://e4ftl01.cr.usgs.gov/MEASURES/GFSAD30AFCE.001/2013.01.01',
                      'CGLC': 'https://s3-eu-west-1.amazonaws.com/vito.landcover.global/v3.0.1'}
        # self.products = ['FROM', 'CCI', 'GFSAD', 'CGLC']
        self.products = config['lc_ref']['products']
        self.years = {product: years[product] for product in self.products}
        self.base_links = {product: base_links[product] for product in self.products}
        self.geom = config['lc_ref']['geom']



def main(config_path):
    # config_path = 'cfgs/config_main_tz_s2.yaml'
    with open(config_path, 'r') as yaml_file:
        config = yaml.load(yaml_file)
