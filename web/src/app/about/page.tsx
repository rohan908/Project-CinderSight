// web/src/app/about/page.tsx
'use client';

import React from 'react';
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
        <p className="text-lg text-gray-600 max-w-6xl mx-auto leading-relaxed">
          CinderSight is a state-of-the-art wildfire spread prediction system that leverages advanced deep learning techniques to enhance wildfire management strategies. With wildfires increasingly threatening ecosystems, infrastructure, and human lives, precise and timely predictions are crucial for effective response and mitigation.
          Our goal is to empower wildfire management agencies, environmental organizations, and emergency responders with an interpretable, highly accurate prediction model, aiming to improve preparedness, help optimize resource allocation, and reduce environmental and economic damage.
        </p>
      </div>

      <Card className="mb-8">
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


