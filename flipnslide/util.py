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
import torch



def download_image(coords, time_range, 
                   verbose:bool=False,
                   **kwargs):
    '''
    Downloads image cube from planetary computer for a given
    set of decimal degree coordinates across a given time frame 
    in a given set of bands in a specified catalog.
    
    Planetary Computer is used as the helper code for 
    downloading COGs. Please check their documentation for 
    catalog choices
    '''
    
    # Define optional params with defaults
    bands = kwargs.get('bands', ['blue', 'green', 'red', 'nir08'])
    cat_name = kwargs.get('cat_name', ['landsat-c2-l2'])
    cloud_cov = kwargs.get('cloud_cov', 5)
    res = kwargs.get('res', 30)
    
    if verbose == True:
        print('Initialize the Planetary Computer connection...')
        
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
    
    if verbose == True:
        print('Search Planetary Computer for requested image/s...')

    #Find items in Planetary Computer within search parameters
    search = catalog.search(collections=cat_name, 
                            bbox=bounds_latlon, 
                            datetime=time_range, 
                            query={'eo:cloud_cover': {'lt': cloud_cov}
                                  })
    
    items = search.item_collection()
    
    #Median stack images by month, constraining resolution
    item = items[0]
    epsg = proj.ext(item).epsg
    
    stack = stackstac.stack(items, epsg=epsg, 
                            assets=bands,
                            bounds_latlon=bounds_latlon, 
                            resolution=res)
    
    #Future functionality will allow for time resolution
    # monthly = stack.resample(time="MS").median("time", keep_attrs=True)
    
    if verbose == True:
        print('Stack image/s across time into one image...')
    
    #For now average across time
    merged = stack.median("time", keep_attrs=True).squeeze().compute()
    
    #Move to a numpy array with correct dimensions
    merged_set = merged.to_dataset(dim='band')
    image_data = merged_set.to_array(dim='band').data
    
    return image_data


def preprocess(data, verbose:bool=False, **kwargs):
    '''
    Preprocesses image cube to prepare for use with ML algorithms.
    
    Fills in any nans band-by-band using scipy interpolation. 
    Then standardizes the data cube, saving the cleaned data.
    '''
    
    #In a future version:
    #Throw out time frames with too many nans
    
    #Fills any nans using scipy interpolate 
    if verbose == True:
        print('Fill in nans...')
    
    band_count = data.shape[0]
    
    for band in range(band_count):
        interp_mask=np.where(~np.isnan(data[band]))
        interp = NearestNDInterpolator(np.transpose(interp_mask), data[band][interp_mask])
        data[band] = interp(*np.indices(data[band].shape))
        
    #!> A good test here is to check that there aren't any nans <!#
        
    #Shift and scale the data cube, band-by-band
    if verbose == True:
        print('Standardize image for use with machine learning...')
        
    normed_data = np.zeros(data.shape)

    for band in range(band_count):
        normed_data[band] = ((data[band] - np.mean(data[band]))/np.std(data[band]))
        
    return normed_data


def saver(tiles,
          file_type:str = 'tensor', 
          save_path:str = '',
          file_name:str = 'tiles',
          **kwargs):
    '''
    Saves data to file, as either numpy ndarrays or pytorch tensors.
    '''
    if file_type == 'tensor':
        torch.save(tiles, f'{save_path}{file_name}.pt')
    
    elif file_type == 'array':
        np.save(f'{save_path}{file_name}.npy', tiles)