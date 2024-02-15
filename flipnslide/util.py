''' General Utilities '''

import pystac_client
from pystac.extensions.projection import ProjectionExtension as proj
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer
import rasterio
import rasterio.features
import stackstac
import xarray as xr
import geopandas as gpd



def download_image(coords, time_range, 
                   **kwargs):
    '''
    Downloads image cube from planetary computer for a given
    set of coordinates, in a given set of bands.
    '''
    
    #Initiate Planetary Computer catalog instance
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
        )
    
    #Define image area to download
    south_bound = coords[0]
    north_boung = coords[1]
    east_bound = coords[2]
    west_bound = coords[3]
    
    area_of_interest = {
        "type": "Polygon",
        "coordinates": [
            [
                #longitude,  latitude
                [west_bound, south_bound],
                [east_bound, south_bound],
                [east_bound, north_boung],
                [west_bound, north_boung],
                [west_bound, south_bound],
            ]
        ],
    }

    bounds_latlon = rasterio.features.bounds(area_of_interest)
    
    #Define other image constraints
    time_range = "2015-08-15/2015-12-31"
    bbox = bounds_latlon
    band_names = ['blue', 'green', 'red', 'nir08']
    cat_name = 'landsat-c2-l2'
    cloud_cov = 5

    #Fill catalog instance
    search = catalog.search(collections=[cat_name], 
                            bbox=bbox, 
                            datetime=time_range, 
                            query={'eo:cloud_cover': {'lt': cloud_cov}
                                  })
    
    items = search.get_all_items()
    
    #Median stack images by month, constraining resolution
    item = lc_items[0]
    lc_epsg = proj.ext(item).epsg
    
    stack = stackstac.stack(items, epsg=lc_epsg, 
                            assets=band_names,
                            bounds_latlon=bounds_latlon, resolution=30)
