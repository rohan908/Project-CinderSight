// web/src/app/about/page.tsx
'use client';

import React from 'react';
import Image from 'next/image';
import { Info, TrendingUp, Cpu, Eye } from 'lucide-react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '@/components/ui/card';

export default function AboutPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="text-center mb-12">
        <div className="flex items-center justify-center mb-4">
          <Info className="w-12 h-12 text-orange-600 mr-3" />
          <h1 className="text-4xl font-bold text-gray-900">
            CinderSight: Wildfire Prediction with AI
          </h1>
        </div>
        <p className="text-xl text-gray-600 max-w-6xl mx-auto leading-relaxed">
          Wildfires result in the emission of nearly 3.3 billion tonnes of CO2 annually and cost the U.S. economy an estimated $70 billion. With a steady increase in wildfires worldwide, emergency responders and wildfire management agencies face significant challenges in allocating resources to mitigate environmental and economic damage. CinderSight is a state-of-the-art wildfire spread prediction system that enhances wildfire management strategies by leveraging advanced deep learning techniques. We aim to deliver precise and timely predictions on wildfire movement to aid in effective response and mitigation. Through CinderSight, we hope to empower emergency responders and wildfire management agencies to mitigate the threat to our ecosystems, infrastructure, and lives.
        </p>
        <p className="mt-4 text-lg text-gray-600 max-w-4xl mx-auto">
          Explore our site for an in-depth look at what powers our model!
        </p>
      </div>

    <Card className="mt-8 mx-auto max-w-xl">
        <CardHeader className = "text-center">
          <CardTitle>Wildfire Risk</CardTitle>
          <CardDescription>
            Forest Service - U.S. Department of Agriculture
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="max-w-full mx-auto relative">
              <Image
                src="/images/wrc-risk.png"
                alt="Map of fire risk"
                width={500}
                height={100}
                className="object-contain rounded-md border mx-auto"
              />
            </div>
        </CardContent>
      </Card>

      <Card className="mb-12 mt-10">
        <CardHeader>
          <CardTitle>Project Objectives</CardTitle>
          <CardDescription>
            Our aims for CinderSight
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="flex flex-col items-start">
              <TrendingUp className="w-6 h-6 text-green-600 mb-2" />
              <h4 className="text-lg font-semibold text-gray-800">
                Enhance Accuracy
              </h4>
              <p className="text-gray-600 text-sm mt-1">
                Integrate enriched multimodal data (satellite imagery, weather forecasts, terrain, and human factors) to boost wildfire spread forecasting performance.
              </p>
            </div>

            <div className="flex flex-col items-start">
              <Cpu className="w-6 h-6 text-blue-600 mb-2" />
              <h4 className="text-lg font-semibold text-gray-800">
                Innovative Architecture
              </h4>
              <p className="text-gray-600 text-sm mt-1">
                Design and implement deep learning models (CNNs, ConvLSTMs, Transformers) tailored for real-time wildfire forecasting.
              </p>
            </div>

            <div className="flex flex-col items-start">
              <Eye className="w-6 h-6 text-indigo-600 mb-2" />
              <h4 className="text-lg font-semibold text-gray-800">
                Interpretability
              </h4>
              <p className="text-gray-600 text-sm mt-1">
                Provide clear, actionable insights so decision-makers and responders can trust and act on predictions.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}


