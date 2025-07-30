"""Enhanced NDWS Dataset Configuration

This module contains all shared constants, feature definitions, and data statistics
for the Enhanced Next Day Wildfire Spread (NDWS) dataset as described in the paper
by Rufai Yusuf Zakari et al.

This configuration is used by both the data visualizer and data cleaner scripts
to ensure consistency across the processing pipeline.
"""

"""Constants for the Enhanced NDWS data reader."""

# Enhanced NDWS Dataset Features (19 total input features)
ENHANCED_INPUT_FEATURES = [
    # Weather factors (current day) - 8 features
    'vs',        # Wind speed (m/s)
    'pr',        # Precipitation (mm)  
    'sph',       # Specific humidity
    'tmmx',      # Max temperature (K)
    'tmmn',      # Min temperature (K)
    'th',        # Wind direction (degrees)
    'erc',       # Energy release component (BTU/sq ft)
    'pdsi',      # Palmer Drought Severity Index
    
    # Weather forecasts (next day) - 4 features
    'ftemp',     # Forecast temperature (K)
    'fpr',       # Forecast precipitation (mm)
    'fws',       # Forecast wind speed (m/s)
    'fwd',       # Forecast wind direction (degrees)
    
    # Terrain factors - 3 features
    'elevation', # Elevation (m)
    'aspect',    # Aspect (degrees)
    'slope',     # Slope (degrees)
    
    # Vegetation - 2 features  
    'ndvi',      # Normalized Difference Vegetation Index
    'evi',       # Enhanced Vegetation Index
    
    # Human factors - 1 feature
    'population', # Population density (people/sq km)
    
    # Fire context - 1 feature
    'prevfiremask' # Previous fire mask
]

# Output features (target variable)
OUTPUT_FEATURES = ['FireMask']

# Legacy input features from original NDWS dataset (for backward compatibility)
LEGACY_INPUT_FEATURES = ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 
                          'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']

# Enhanced Data Statistics for normalization and clipping
# For each variable, the statistics are ordered in the form:
# (min_clip, max_clip, mean, standard_deviation)
ENHANCED_DATA_STATS = {
    # Weather factors (current day)
    
    # Wind speed in m/s.
    # Negative values do not make sense, given there is a wind direction.
    # 0., 99.9 percentile
    'vs': (0.0, 10.024310074806237, 3.8500874, 1.4109988),
    
    # Precipitation in mm.
    # Negative values do not make sense, so min is set to 0.
    # 0., 99.9 percentile
    'pr': (0.0, 44.53038024902344, 1.7398051, 4.482833),
    
    # Specific humidity.
    # Negative values do not make sense, so min is set to 0.
    # The range of specific humidity is up to 100% so max is 1.
    'sph': (0., 1., 0.0071658953, 0.0042835088),
    
    # Max temperature in Kelvin.
    # -20 degree C, 99.9 percentile
    'tmmx': (253.15, 315.09228515625, 295.17383, 9.815496),
    
    # Min temperature in Kelvin.
    # -20 degree C, 99.9 percentile
    'tmmn': (253.15, 298.94891357421875, 281.08768, 8.982386),
    
    # Wind direction in degrees clockwise from north.
    # Thus min set to 0 and max set to 360.
    'th': (0., 360.0, 190.32976, 72.59854),
    
    # NFDRS fire danger index energy release component expressed in BTU's per
    # square foot.
    # Negative values do not make sense. Thus min set to zero.
    # 0., 99.9 percentile
    'erc': (0.0, 106.24891662597656, 37.326267, 20.846027),
    
    # Drought Index (Palmer Drought Severity Index)
    # 0.1 percentile, 99.9 percentile
    'pdsi': (-6.12974870967865, 7.876040384292651, -0.0052714925, 2.6823447),
    
    # Weather forecasts (next day) - estimated based on current weather patterns
    
    # Forecast temperature in Kelvin.
    # Based on temperature range similar to current day temperatures
    'ftemp': (253.15, 315.0, 288.0, 10.0),
    
    # Forecast precipitation in mm.
    # Similar range to current day precipitation
    'fpr': (0.0, 45.0, 1.8, 4.5),
    
    # Forecast wind speed in m/s.
    # Similar range to current day wind speed
    'fws': (0.0, 10.0, 3.9, 1.4),
    
    # Forecast wind direction in degrees clockwise from north.
    # Full range 0-360 degrees
    'fwd': (0., 360.0, 180.0, 90.0),
    
    # Terrain factors
    
    # Elevation in m.
    # 0.1 percentile, 99.9 percentile
    'elevation': (0.0, 3141.0, 657.3003, 649.0147),
    
    # Aspect in degrees.
    # Slope aspect from 0 to 360 degrees
    'aspect': (0.0, 360.0, 180.0, 90.0),
    
    # Slope in degrees.
    # Terrain slope, typically 0-45 degrees for most terrain
    'slope': (0.0, 45.0, 5.0, 8.0),
    
    # Vegetation indices
    
    # Vegetation index (NDVI times 10,000, since it's supposed to be b/w -1 and 1)
    # min, max values from dataset analysis
    'NDVI': (-9821.0, 9996.0, 5157.625, 2466.6677),
    
    # Enhanced Vegetation Index (EVI)
    # Similar scaling to NDVI but with enhanced sensitivity
    'EVI': (-2000.0, 10000.0, 3000.0, 2000.0),
    
    # Human factors
    
    # Population density in people per square km.
    # min, 99.9 percentile
    'population': (0., 2534.06298828125, 25.531384, 154.72331),
    
    # Fire context - We don't want to normalize the FireMasks.
    # 1 indicates fire, 0 no fire, -1 unlabeled data
    'PrevFireMask': (-1., 1., 0., 1.),
    'FireMask': (-1., 1., 0., 1.)
}

"""Feature groupings for analysis and visualization."""

# Weather features (current day)
WEATHER_CURRENT_FEATURES = ['vs', 'pr', 'sph', 'tmmx', 'tmmn', 'th', 'erc', 'pdsi']

# Weather forecast features (next day)
WEATHER_FORECAST_FEATURES = ['ftemp', 'fpr', 'fws', 'fwd']

# Terrain features
TERRAIN_FEATURES = ['elevation', 'aspect', 'slope']

# Vegetation features
VEGETATION_FEATURES = ['NDVI', 'EVI']

# Human factors
HUMAN_FEATURES = ['population']

# Fire context features
FIRE_FEATURES = ['PrevFireMask']

"""Dataset metadata and processing parameters."""

# Default data dimensions
DEFAULT_DATA_SIZE = 64
DEFAULT_SAMPLE_SIZE = 32

# Number of input and output channels
NUM_ENHANCED_INPUT_FEATURES = len(ENHANCED_INPUT_FEATURES)
NUM_OUTPUT_CHANNELS = len(OUTPUT_FEATURES)

# Data processing constants
INVALID_DATA_VALUE = -1.0
EPSILON = 1e-6  # Small value to avoid division by zero

# Feature descriptions and units from the enhanced dataset paper
FEATURE_DESCRIPTIONS = {
    'vs': 'Wind Speed (m/s)',
    'pr': 'Precipitation (mm/day)',
    'sph': 'Specific Humidity (kg/kg)',
    'tmmx': 'Maximum Temperature (°C)',
    'tmmn': 'Minimum Temperature (°C)',
    'th': 'Wind Direction (degrees)',
    'erc': 'Energy Release Component (unitless)',
    'pdsi': 'Palmer Drought Severity Index (unitless)',
    'ftemp': 'Forecast Temperature (°C)',
    'fpr': 'Forecast Precipitation (mm/day)',
    'fws': 'Forecast Wind Speed (m/s)',
    'fwd': 'Forecast Wind Direction (degrees)',
    'elevation': 'Elevation (meters)',
    'aspect': 'Aspect (degrees)',
    'slope': 'Slope (degrees)',
    'ndvi': 'Normalized Difference Vegetation Index (unitless)',
    'evi': 'Enhanced Vegetation Index (unitless)',
    'population': 'Population Density (people/km²)',
    'prevfiremask': 'Previous Day Fire Mask (binary)'
}

# Feature categories for better organization
FEATURE_CATEGORIES = {
    'weather_current': {
        'features': ['vs', 'pr', 'sph', 'tmmx', 'tmmn', 'th', 'erc', 'pdsi'],
        'description': 'Current Day Weather Factors',
        'colormap': 'viridis'
    },
    'weather_forecast': {
        'features': ['ftemp', 'fpr', 'fws', 'fwd'],
        'description': 'Next Day Weather Forecast',
        'colormap': 'plasma'
    },
    'terrain': {
        'features': ['elevation', 'aspect', 'slope'],
        'description': 'Terrain Factors',
        'colormap': 'terrain'
    },
    'vegetation': {
        'features': ['ndvi', 'evi'],
        'description': 'Vegetation Indices',
        'colormap': 'Greens'
    },
    'human': {
        'features': ['population'],
        'description': 'Human Factors',
        'colormap': 'Blues'
    },
    'fire': {
        'features': ['prevfiremask'],
        'description': 'Fire History',
        'colormap': 'Reds'
    }
} 