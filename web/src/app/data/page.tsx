// web/src/app/data/page.tsx
'use client';

import React from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { Database, CloudRain, Map, Users } from 'lucide-react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '@/components/ui/card';

export default function DataPage() {
  const DATASET_URL =
    'https://www.kaggle.com/datasets/rufaiyusufzakari/enhanced-and-modified-next-day-wildfire-spread';

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <div className="flex items-center justify-center mb-4">
          <Database className="w-12 h-12 text-green-600 mr-3" />
          <h1 className="text-4xl font-bold text-gray-900">
            Comprehensive &amp; Enhanced Wildfire Dataset
          </h1>
        </div>
        <p className="text-xl text-gray-600 max-w-6xl mx-auto">
          The CinderSight project utilizes an enriched version of the Next Day Wildfire Spread (NDWS) dataset, expanding from 12 to 19 features for comprehensive wildfire modeling.
        </p>
        <p className="text-xl text-gray-600 max-w-6xl mx-auto">
          The enhanced dataset spans from July 2015 to October 2024, covering 39,333 samples across the contiguous United States at 1 km spatial resolution. It incorporates next-day weather forecasts and terrain features to provide substantial detail for wildfire spread modeling.
        </p>
        <p className="mt-4 text-lg text-gray-600 max-w-4xl mx-auto">
          Download the full enriched NDWS dataset{' '}
          <Link
            href={DATASET_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="underline text-blue-600 hover:text-blue-800"
          >
            here
          </Link>
          .
        </p>
      </div>


      <Card className="mt-8 mb-10">
        <CardHeader className="text-center">
          <CardTitle>Sample Feature &amp; Mask Grid</CardTitle>
          <CardDescription>
            A quick glance at all input features plus fire masks.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="max-w-full mx-auto relative">
              <Image
                src="/images/feature-grid.png"
                alt="Feature grid showing elevation, slope, NDVI, fire mask, etc."
                width={1300}
                height={325}
                className="object-contain rounded-md border"
              />
            </div>
        </CardContent>
      </Card>



      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Enhanced Dataset Features (19 Total)</CardTitle>
          <CardDescription>
            Comprehensive environmental, meteorological, and anthropogenic factors
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Weather Factors */}
            <div className="space-y-4">
              <div className="flex items-center mb-3">
                <CloudRain className="w-6 h-6 text-blue-600 mr-2" />
                <h4 className="text-lg font-semibold text-gray-800">Weather Factors (Current Day) - 8 Features</h4>
              </div>
              <div className="space-y-2 text-sm text-gray-600">
                <p>• <strong>Wind Speed (vs)</strong>: Daily wind speed in m/s from GRIDMET (4 km resolution)</p>
                <p>• <strong>Precipitation (pr)</strong>: Daily precipitation in mm/day from GRIDMET (4 km resolution)</p>
                <p>• <strong>Specific Humidity (sph)</strong>: Daily specific humidity in kg/kg from GRIDMET (4 km resolution)</p>
                <p>• <strong>Max Temperature (tmmx)</strong>: Daily maximum temperature in °C from GRIDMET (4 km resolution)</p>
                <p>• <strong>Min Temperature (tmmn)</strong>: Daily minimum temperature in °C from GRIDMET (4 km resolution)</p>
                <p>• <strong>Wind Direction (th)</strong>: Daily wind direction in degrees from GRIDMET (4 km resolution)</p>
                <p>• <strong>Energy Release Component (erc)</strong>: Daily ERC from NFDRS (1 km resolution)</p>
                <p>• <strong>Drought Index (pdsi)</strong>: Palmer Drought Severity Index from GRIDMET (4 km resolution)</p>
              </div>
            </div>

            {/* Weather Forecast */}
            <div className="space-y-4">
              <div className="flex items-center mb-3">
                <CloudRain className="w-6 h-6 text-cyan-600 mr-2" />
                <h4 className="text-lg font-semibold text-gray-800">Weather Forecast (Next Day) - 4 Features</h4>
              </div>
              <div className="space-y-2 text-sm text-gray-600">
                <p>• <strong>Forecast Temperature (ftemp)</strong>: Next-day temperature forecast in °C from NOAA GFS (27.83 km resolution)</p>
                <p>• <strong>Forecast Precipitation (fpr)</strong>: Next-day precipitation forecast in mm/day from NOAA GFS (27.83 km resolution)</p>
                <p>• <strong>Forecast Wind Speed (fws)</strong>: Next-day wind speed forecast in m/s from NOAA GFS (27.83 km resolution)</p>
                <p>• <strong>Forecast Wind Direction (fwd)</strong>: Next-day wind direction forecast in degrees from NOAA GFS (27.83 km resolution)</p>
              </div>
            </div>

            {/* Terrain Factors */}
            <div className="space-y-4">
              <div className="flex items-center mb-3">
                <Map className="w-6 h-6 text-green-600 mr-2" />
                <h4 className="text-lg font-semibold text-gray-800">Terrain Factors - 3 Features</h4>
              </div>
              <div className="space-y-2 text-sm text-gray-600">
                <p>• <strong>Elevation</strong>: Terrain elevation in meters from SRTM (30 m resolution, downsampled to 1 km)</p>
                <p>• <strong>Aspect</strong>: Terrain aspect in degrees from SRTM (30 m resolution, downsampled to 1 km)</p>
                <p>• <strong>Slope</strong>: Terrain slope in degrees from SRTM (30 m resolution, downsampled to 1 km)</p>
              </div>
            </div>

            {/* Vegetation & Human Factors */}
            <div className="space-y-4">
              <div className="flex items-center mb-3">
                <Database className="w-6 h-6 text-purple-600 mr-2" />
                <h4 className="text-lg font-semibold text-gray-800">Vegetation & Human Factors - 3 Features</h4>
              </div>
              <div className="space-y-2 text-sm text-gray-600">
                <p>• <strong>NDVI</strong>: Normalized Difference Vegetation Index from VIIRS (500 m resolution, downsampled to 1 km)</p>
                <p>• <strong>EVI</strong>: Enhanced Vegetation Index from VIIRS (500 m resolution, downsampled to 1 km)</p>
                <p>• <strong>Population Density</strong>: Population density in people/km² from GPWv4 (927.67 m resolution, downsampled to 1 km)</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Data Processing Pipeline</CardTitle>
          <CardDescription>
            Comprehensive preprocessing steps for high-quality model inputs
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800 mb-3">Spatial & Temporal Alignment</h4>
              <div className="space-y-2 text-sm text-gray-600">
                <p>• <strong>Spatial Resolution</strong>: All data aligned to 1 km resolution to match fire mask resolution</p>
                <p>• <strong>Topography & Vegetation</strong>: Downsampled using bicubic interpolation</p>
                <p>• <strong>Weather & Drought</strong>: Upsampled using bicubic interpolation</p>
                <p>• <strong>Temporal Alignment</strong>: Hourly forecasts aggregated to daily means</p>
                <p>• <strong>Vegetation Composites</strong>: 16-day composites linearly interpolated to daily values</p>
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800 mb-3">Data Augmentation & Spatial Processing</h4>
              <div className="space-y-2 text-sm text-gray-600">
                <p>• <strong>Value Clipping</strong>: Features clipped to remove extreme values (0.1st to 99.9th percentiles)</p>
                <p>• <strong>Normalization</strong>: Mean and standard deviation normalization from training dataset</p>
                <p>• <strong>Random Cropping</strong>: 32×32 km regions randomly cropped from 64×64 km regions</p>
                <p>• <strong>Rotation</strong>: Each sample rotated by 0°, 90°, 180°, and 270° for data variability</p>
                <p>• <strong>Center Cropping</strong>: Focused analysis since fires are generally centered</p>
                <p>• <strong>Spatial Context Processing</strong>: Each pixel's 19 features are expanded to 171 features by incorporating surrounding 8 neighboring pixels (9 total positions × 19 features = 171 channels)</p>
                <p>• <strong>Surrounding Position Function</strong>: Creates spatial context by concatenating center pixel with up, down, left, right, and diagonal neighbors</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Dataset Statistics</CardTitle>
          <CardDescription>
            Key metrics and distribution information
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <h4 className="text-2xl font-bold text-blue-600">39,333</h4>
              <p className="text-sm text-gray-600">Total Samples</p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <h4 className="text-2xl font-bold text-green-600">1 km</h4>
              <p className="text-sm text-gray-600">Spatial Resolution</p>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <h4 className="text-2xl font-bold text-purple-600">64×64</h4>
              <p className="text-sm text-gray-600">Input Resolution (pixels)</p>
            </div>
            <div className="text-center p-4 bg-orange-50 rounded-lg">
              <h4 className="text-2xl font-bold text-orange-600">2015-2024</h4>
              <p className="text-sm text-gray-600">Temporal Coverage</p>
            </div>
          </div>
          
          <div className="mt-6">
            <h4 className="text-lg font-semibold text-gray-800 mb-3">Class Distribution</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-3 bg-red-50 rounded-lg">
                <h5 className="text-lg font-semibold text-red-600">1.34%</h5>
                <p className="text-sm text-gray-600">Fire Pixels</p>
              </div>
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <h5 className="text-lg font-semibold text-gray-600">97.81%</h5>
                <p className="text-sm text-gray-600">No-Fire Pixels</p>
              </div>
              <div className="text-center p-3 bg-yellow-50 rounded-lg">
                <h5 className="text-lg font-semibold text-yellow-600">0.85%</h5>
                <p className="text-sm text-gray-600">Unlabeled Pixels</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Feature Statistics</CardTitle>
          <CardDescription>
            Statistical summary of all 19 features after preprocessing
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2 font-semibold">Feature</th>
                  <th className="text-right p-2 font-semibold">Mean</th>
                  <th className="text-right p-2 font-semibold">Std</th>
                  <th className="text-right p-2 font-semibold">Min</th>
                  <th className="text-right p-2 font-semibold">Max</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Wind Speed (vs)</td>
                  <td className="text-right p-2">3.529</td>
                  <td className="text-right p-2">0.886</td>
                  <td className="text-right p-2">1.521</td>
                  <td className="text-right p-2">6.660</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Precipitation (pr)</td>
                  <td className="text-right p-2">0.206</td>
                  <td className="text-right p-2">1.095</td>
                  <td className="text-right p-2">-0.163</td>
                  <td className="text-right p-2">17.820</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Specific Humidity (sph)</td>
                  <td className="text-right p-2">0.006</td>
                  <td className="text-right p-2">0.003</td>
                  <td className="text-right p-2">0.002</td>
                  <td className="text-right p-2">0.014</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Max Temperature (tmmx)</td>
                  <td className="text-right p-2">299.411</td>
                  <td className="text-right p-2">6.743</td>
                  <td className="text-right p-2">280.763</td>
                  <td className="text-right p-2">311.845</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Min Temperature (tmmn)</td>
                  <td className="text-right p-2">282.481</td>
                  <td className="text-right p-2">6.785</td>
                  <td className="text-right p-2">267.686</td>
                  <td className="text-right p-2">295.878</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Wind Direction (th)</td>
                  <td className="text-right p-2">229.476</td>
                  <td className="text-right p-2">76.533</td>
                  <td className="text-right p-2">24.903</td>
                  <td className="text-right p-2">339.065</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Energy Release Component (erc)</td>
                  <td className="text-right p-2">55.302</td>
                  <td className="text-right p-2">26.148</td>
                  <td className="text-right p-2">10.002</td>
                  <td className="text-right p-2">103.569</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Drought Index (pdsi)</td>
                  <td className="text-right p-2">-0.113</td>
                  <td className="text-right p-2">3.063</td>
                  <td className="text-right p-2">-7.003</td>
                  <td className="text-right p-2">8.182</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Forecast Temperature (ftemp)</td>
                  <td className="text-right p-2">24.102</td>
                  <td className="text-right p-2">5.379</td>
                  <td className="text-right p-2">8.118</td>
                  <td className="text-right p-2">35.256</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Forecast Precipitation (fpr)</td>
                  <td className="text-right p-2">0.007</td>
                  <td className="text-right p-2">0.002</td>
                  <td className="text-right p-2">0.002</td>
                  <td className="text-right p-2">0.017</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Forecast Wind Speed (fws)</td>
                  <td className="text-right p-2">-0.145</td>
                  <td className="text-right p-2">2.860</td>
                  <td className="text-right p-2">-6.521</td>
                  <td className="text-right p-2">7.361</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Forecast Wind Direction (fwd)</td>
                  <td className="text-right p-2">2.647</td>
                  <td className="text-right p-2">3.084</td>
                  <td className="text-right p-2">-4.446</td>
                  <td className="text-right p-2">13.176</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Elevation</td>
                  <td className="text-right p-2">959.124</td>
                  <td className="text-right p-2">901.733</td>
                  <td className="text-right p-2">0.000</td>
                  <td className="text-right p-2">4054.000</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Aspect</td>
                  <td className="text-right p-2">169.162</td>
                  <td className="text-right p-2">96.550</td>
                  <td className="text-right p-2">0.000</td>
                  <td className="text-right p-2">359.850</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Slope</td>
                  <td className="text-right p-2">2.694</td>
                  <td className="text-right p-2">3.766</td>
                  <td className="text-right p-2">0.000</td>
                  <td className="text-right p-2">35.420</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">NDVI</td>
                  <td className="text-right p-2">4926.497</td>
                  <td className="text-right p-2">1438.924</td>
                  <td className="text-right p-2">1809.648</td>
                  <td className="text-right p-2">8463.683</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">EVI</td>
                  <td className="text-right p-2">2731.871</td>
                  <td className="text-right p-2">784.070</td>
                  <td className="text-right p-2">1233.121</td>
                  <td className="text-right p-2">5540.832</td>
                </tr>
                <tr className="border-b hover:bg-gray-50">
                  <td className="p-2 font-medium">Population Density</td>
                  <td className="text-right p-2">13.811</td>
                  <td className="text-right p-2">103.098</td>
                  <td className="text-right p-2">0.000</td>
                  <td className="text-right p-2">5283.745</td>
                </tr>
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Feature Correlation Matrix</CardTitle>
          <CardDescription>
            Pearson correlation coefficients between all 19 features
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex justify-center items-center">
            <Image
              src="/images/feature_correlations.png"
              alt="Feature correlation matrix heatmap showing relationships between all 19 features"
              width={800}
              height={800}
              className="object-contain rounded-md border"
            />
          </div>
          <div className="mt-4 text-sm text-gray-600">
            <h4 className="font-semibold mb-2">Key Correlations:</h4>
            <ul className="space-y-1">
              <li>• <strong>Temperature Correlation</strong>: Max and min temperatures show very high correlation (0.93)</li>
              <li>• <strong>Vegetation Indices</strong>: NDVI and EVI are highly correlated (0.90) as both measure vegetation greenness</li>
              <li>• <strong>Drought & Energy</strong>: PDSI and ERC show strong negative correlation (-0.73), indicating drought increases fire risk</li>
              <li>• <strong>Elevation & Fire Risk</strong>: Elevation and ERC show strong positive correlation (0.77)</li>
              <li>• <strong>Weather Forecasts</strong>: Forecast temperature strongly correlates with current temperatures (0.87)</li>
            </ul>
          </div>
        </CardContent>
      </Card>



    </div>
  );
}






