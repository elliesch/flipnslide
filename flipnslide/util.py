''' General Utilities '''

import pystac_client
from pystac.extensions.projection import ProjectionExtension as proj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer
import rasterio
import rasterio.features
import stackstac
import xarray as xr
import geopandas as gpd
from scipy.interpolate import NearestNDInterpolator



def download_image(coords, time_range, bands, cat_name,
                   cloud_cov, res, **kwargs):
    '''
    Downloads image cube from planetary computer for a given
    set of decimal degree coordinates across a given time frame 
    in a given set of bands in a specified catalog.
    
    Planetary Computer is used as the helper code for 
    downloading COGs. Please check their documentation for 
    catalog choices
    '''
    
    #Initiate Planetary Computer catalog instance
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
        )
    
    #Define image area to download
    south_bound = coords[0]
    north_bound = coords[1]
    east_bound = coords[2]
    west_bound = coords[3]
    
    area_of_interest = {
        "type": "Polygon",
        "coordinates": [
            [
                #longitude,  latitude
                [west_bound, south_bound],
                [east_bound, south_bound],
                [east_bound, north_bound],
                [west_bound, north_bound],
                [west_bound, south_bound],
            ]
        ],
    }

    bounds_latlon = rasterio.features.bounds(area_of_interest)

    #Find items in Planetary Computer within search parameters
    search = catalog.search(collections=cat_name, 
                            bbox=bounds_latlon, 
                            datetime=time_range, 
                            query={'eo:cloud_cover': {'lt': cloud_cov}
                                  })
    
    items = search.get_all_items()
    
    #Median stack images by month, constraining resolution
    item = lc_items[0]
    lc_epsg = proj.ext(item).epsg
    
    stack = stackstac.stack(items, epsg=lc_epsg, 
                            assets=bands,
                            bounds_latlon=bounds_latlon, 
                            resolution=res)
    
    monthly = stack.resample(time="MS").median("time", keep_attrs=True)
    
    merged = stackstac.mosaic(stack, dim="time", axis=None).squeeze().compute()
    
    #Move to a numpy array with correct dimensions
    merged_set = merged.to_dataset(dim='band')
    image_data = merged_set.to_array(dim='band').data
    
    return image_data


def preprocess(data, **kwargs):
    '''
    Preprocesses image cube to prepare for use with ML algorithms.
    
    Fills in any nans band-by-band using scipy interpolation. 
    Then standardizes the data cube, saving the cleaned data.
    '''
    
    #Throw out time frames with too many nans
    ###!!!>>> HAVE TO FIGURE OUT WHAT DIMS WOULD BE TIME <<<!!!###
    
    #Fills any nans using scipy interpolate 
    ###!!!>>> DOUBLE CHECK THAT THIS SHAPE USING CORRECT DIM <<<!!!###
    band_count = data.shape[0]
    
    for band in range(band_count):
        interp_mask=np.where(~np.isnan(data[band]))
        interp = NearestNDInterpolator(np.transpose(interp_mask), data[band][interp_mask])
        data[band] = interp(*np.indices(data[band].shape))
        
    #!> A good test here is to check that there aren't any nans <!#
        
    #Shift and scale the data cube, band-by-band
    ###!!!>>> DOUBLE CHECK THAT BAND_COUNT USING CORRECT DIM <<<!!!###
    normed_data = np.zeros(data.shape)

    for band in range(band_count):
        normed_data[band] = ((data[band] - np.mean(data[band]))/np.std(data[band]))
        
    return normed_data