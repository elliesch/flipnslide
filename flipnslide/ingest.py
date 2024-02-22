''' Ingest Images (right now only optimized for downloading COGs) '''

from .util import download_image, preprocess


class ImageIngest:
    
    def __init__(self, coords, time_range, 
                 bands:list=['blue', 'green', 'red', 'nir08'], 
                 cat_name:list=['landsat-c2-l2'],
                 cloud_cov:int=5,
                 res:int=30, **kwargs):
        
        #Meta data
        self.coords = coords
        self.time_range = time_range
        self.bands = bands
        self.res = res
        
        #Import image and preprocess
        raw_image = download_image(coords, time_range, bands, 
                                   cat_name, cloud_cov, res)
        
        self.image = preprocess(raw_image)