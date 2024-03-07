''' Ingest Images from Planetary Computer'''

from .util import download_image, preprocess
from .viz import ingest_viz



class ImageIngest:
    
    def __init__(self, coords, time_range, 
                 viz:bool=False, verbose:bool=False, 
                 **kwargs):
        '''
        Initialize ImageIngest to download and preprocess an image.

        Parameters:
        - coords (List[float]): List of four floats indicating corners of the requested image 
          in long/lat coordinates. Should follow this format: 
          [southern_boundary, northern_boundary, eastern_boundary, western_boundary].
        - time_range (str): String indicating the time range for the requested image. 
          Should follow this format: 'YYYY-MM-DD/YYYY-MM-DD'.
        - viz (bool): Flag indicating whether to visualize downloaded image.
          
        Optional Parameters passed from **kwargs:
        - bands (List[str]): List indicating bands of the requested image 
          (default is ['blue', 'green', 'red', 'nir08']).
        - res (int): Integer representing the resolution of the requested image. Should match the 
          resolution of the data catalog (default is 30).

        Attributes:
        - coords (List[float]): List of four floats indicating corners of the requested image in 
          long/lat coordinates.
        - time_range (str): String indicating the time range for the requested image.
        - image (numpy.ndarray): NumPy ndarray representing the preprocessed image.
        
        Optional Attributes:
        - bands (List[str]): List indicating bands of the requested image.
        - res (int): Integer representing the resolution of the requested image.

        Notes:
        - The ImageIngest class is used to download and preprocess an image based on the provided 
          coordinates, time range, and optional parameters.
        - The downloaded image is preprocessed using the preprocess function before being stored in 
          the 'image' attribute.
        '''
        
        #Meta data
        self.coords = coords
        self.time_range = time_range
        
        #Optional meta data
        if 'bands' in kwargs:
            self.bands = kwargs['bands']
            
        if 'res' in kwargs:
            self.res = kwargs['res']
        
        #Import image and preprocess
        if verbose == True:
            raw_image = download_image(coords, time_range, 
                                       verbose=True, **kwargs)
            
        else:
            raw_image = download_image(coords, time_range, **kwargs)
        
        self.image = preprocess(raw_image)
        
        #Visualize the preprocessed image
        if viz == True or verbose == True:
            ingest_viz(self.image)
            