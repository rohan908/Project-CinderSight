from pyproj import Transformer
import rasterio

if __name__ == "__main__":
    # Example usage: Get the coordinates of a specific pixel
    row, col = 40, 40

    with rasterio.open('data/rasters/2023_fireDOYrasters/2023_3_krig.tif') as src:
        data = src.read(1).astype(float)
        transform = src.transform
        crs = src.crs
        lon, lat = src.xy(row, col)  # Get the coordinates of the pixel at (row, col)
        print(f'Pixel at row {row}, col {col} is located at (Lat: {lat}, Lon: {lon})')
        print(f'Transform: {transform}')
        print(f'Coordinate Reference System: {crs}')

        # The coordinate reference system (CRS) used in the CSV files for the geographic coordinates.
        # This is set to WGS 84 (EPSG:4269), which is specified in the project documentation (https://osf.io/f48ry/wiki/home/)
        CSV_CRS = 'EPSG:4269'

        # Create a transformer from the raster's CRS to EPSG:4269
        transformer = Transformer.from_crs(crs, CSV_CRS, always_xy=True)
        standard_lon, standard_lat = transformer.transform(lon, lat)
        print(f'Pixel at row {row}, col {col} in {CSV_CRS} is located at (Lat: {standard_lat}, Lon: {standard_lon})')

