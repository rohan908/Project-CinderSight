// web/src/app/model/page.tsx
'use client';

import React from 'react';
import { Layers, Cpu, Clock, Zap } from 'lucide-react';
import Image from 'next/image';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '@/components/ui/card';

export default function ModelPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="text-center mb-12">
        <div className="flex items-center justify-center mb-4">
          <Layers className="w-12 h-12 text-purple-600 mr-3" />
          <h1 className="text-4xl font-bold text-gray-900">
            Model Architecture: Spatio-temporal to Spatial Transition
          </h1>
        </div>
        <p className="text-lg text-gray-600 max-w-6xl mx-auto leading-relaxed">
          Initially, our model utilized a spatio-temporal architecture, processing data across multiple time steps—days, height, width, and features. It comprised four key components: a CNN-based spatial feature extractor, a predictive block for temporal progression, a Transformer module for capturing long-range dependencies, and a decoder for generating final fire spread masks.
          To optimize efficiency and computational simplicity, we transitioned to a streamlined spatial architecture. This design retains the core CNN-based extractor and predictive block for next-day wildfire forecasting, while removing the explicit temporal Transformer and decoder modules. The result is a more compact, high-performance model tailored for real-time wildfire spread prediction.
        </p>

      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Model Architecture</CardTitle>
          <CardDescription>
            Diagram of our final prediction model architecture
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="max-w-full mx-auto relative">
              <Image
                src="/images/model-diagram.png"
                alt="Diagram showing the architecture of the prediction model"
                width={1300}
                height={325}
                className="object-contain rounded-md border"
              />
            </div>
        </CardContent>
      </Card>


      </div>

      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Key Components</CardTitle>
          <CardDescription>
            Building blocks of our current model
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="flex flex-col items-start">
              <Cpu className="w-6 h-6 text-blue-700 mb-2" />
              <h4 className="text-lg font-semibold text-gray-800">
                CNNModel
              </h4>
              <p className="text-gray-600 text-sm mt-1">
                Extracts spatial features (terrain, vegetation, weather layers) using convolutional layers and positional encodings.
              </p>
            </div>

            <div className="flex flex-col items-start">
              <Clock className="w-6 h-6 text-brown-700 mb-2" />
              <h4 className="text-lg font-semibold text-gray-800">
                NextFramePredictor
              </h4>
              <p className="text-gray-600 text-sm mt-1">
                Predicts next‐day fire spread probabilities from extracted spatial features via ConvLSTM.
              </p>
            </div>

            <div className="flex flex-col items-start">
              <Zap className="w-6 h-6 text-yellow-700 mb-2" />
              <h4 className="text-lg font-semibold text-gray-800">
                Simplified Pipeline
              </h4>
              <p className="text-gray-600 text-sm mt-1">
                Eliminates temporal Transformer and decoder for faster inference and reduced parameter count, without sacrificing accuracy.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}