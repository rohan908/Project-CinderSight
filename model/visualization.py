import numpy as np
import rasterio
import matplotlib.pyplot as plt

# 1. Open the raster
with rasterio.open('data/rasters/2023_fireDOYrasters/2023_100_krig.tif') as src:
    data = src.read(1).astype(float)
    nodata = src.nodata

# 2. Mask out no‑data values
masked = np.ma.masked_equal(data, nodata)

# 3. Plot with a continuous colormap
plt.figure(figsize=(8,6))
im = plt.imshow(
    masked,
    cmap='plasma',       # 'inferno', 'viridis', 'plasma', etc.
    interpolation='none'
)
cbar = plt.colorbar(im, shrink=0.7)
cbar.set_label('Day of Burning')
plt.title('Fire Spread Day-of-Burning Raster')
plt.axis('off')
plt.show()
