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
          The CinderSight project utilizes an enriched version of the Next Day
          Wildfire Spread (NDWS) dataset, tailored specifically for wildfire
          spread modeling across the contiguous United States from July 2015 to
          October 2024. This dataset integrates 19 key environmental factors including
          weather and geographical features, providing great detail
          for wildfire prediction.
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
          <CardTitle>Dataset Highlights</CardTitle>
          <CardDescription>
            Key components of our enriched wildfire spread dataset
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="flex flex-col items-start">
              <Database className="w-6 h-6 text-gray-700 mb-2" />
              <h4 className="text-lg font-semibold text-gray-800">
                Historical Data
              </h4>
              <p className="text-gray-600 text-sm mt-1">
                Daily fire masks from NASA MOD14A1 at 1 km resolution.
              </p>
            </div>

            <div className="flex flex-col items-start">
              <CloudRain className="w-6 h-6 text-blue-700 mb-2" />
              <h4 className="text-lg font-semibold text-gray-800">
                Weather Conditions
              </h4>
              <p className="text-gray-600 text-sm mt-1">
                • GRIDMET current: temperature, humidity, wind speed, precipitation.<br/>
              </p>
            </div>

            <div className="flex flex-col items-start">
              <Map className="w-6 h-6 text-brown-700 mb-2" />
              <h4 className="text-lg font-semibold text-gray-800">
                Terrain &amp; Vegetation
              </h4>
              <p className="text-gray-600 text-sm mt-1">
                • Elevation, slope, aspect from SRTM.<br/>
                • NDVI &amp; EVI indices from NASA VIIRS.
              </p>
            </div>

            <div className="flex flex-col items-start">
              <Users className="w-6 h-6 text-purple-700 mb-2" />
              <h4 className="text-lg font-semibold text-gray-800">
                Human Factors
              </h4>
              <p className="text-gray-600 text-sm mt-1">
                Population density from CIESIN GPWv4.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>



    </div>
  );
}






