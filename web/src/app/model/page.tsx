// web/src/app/model/page.tsx
'use client';

import React from 'react';
import { Layers, ArrowBigRightDash, ZoomIn, MapPinned } from 'lucide-react';
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
            Model Architecture
          </h1>
        </div>
        <div className="text-xl text-gray-600 max-w-6xl mx-auto leading-relaxed space-y-4">
          <p>
            To provide actionable insights to emergency responders and wildfire management services, we needed a robust model architecture to produce clear and precise wildfire spread predictions. Taking inspiration from other models, we first developed a spatiotemporal architecture that processed the 19 environmental covariates from the enriched NDWS dataset, leveraging CNNs for spatial feature extraction, and Transformers and ConvLSTMs to capture temporal dynamics. But due to insufficient sequential data in the enriched NDWS dataset, the temporal components of the architecture were dropped after primary experimentation.
          </p>
          <p className="mb-6">
          The final model employs two components, the <strong>CNNModel</strong> that extracts spatial features and the <strong>NextFramePredictor</strong> that predicts the next day&apos;s wildfire spread map from the extracted spatial features. <strong>Important:</strong> The original 64×64 pixel data is center-cropped to 32×32 pixels before being input into the model to match the training configuration.
          </p>
        </div>
      </div>

      <Card className="mb-8 mt-8">
        <CardHeader>
          <CardTitle>How CinderSight Works</CardTitle>
          <CardDescription>
            Detailed architecture overview
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="flex flex-col items-start">
              <MapPinned className="w-6 h-6 text-red-700 mb-2" />
              <h4 className="text-lg font-semibold text-gray-800">
                Spatial Feature Extraction (CNNModel)
              </h4>
                <p><strong>Input:</strong> 32×32×171 feature maps (19 original features × 9 spatial positions from surrounding pixels)</p>
                <p>Composed of two parallel feature extraction paths:</p>
                <ul className="list-disc list-inside pl-5 space-y-1 text-left">
                  <p><strong>Local Branch:</strong></p>
                  <ul className="list-disc list-inside pl-8 space-y-1">
                    <li>Preserves spatial resolution for detailed feature extraction.</li>
                    <li>Applies convolutional blocks (Conv2DBlock) sequentially, gradually increasing channels.</li>
                  </ul>
                  <p><strong>Global Branch:</strong></p>
                  <ul className="list-disc list-inside pl-8 space-y-1">
                    <li>Captures broader contextual information through downsampling (MaxPool2D) and upsampling (ConvTranspose2d).</li>
                    <li>Uses skip connections to merge downsampled and upsampled features at multiple scales.</li>
                    <li>Ends with a bottleneck, then progressively upscales the data back to the original resolution.</li>
                  </ul>
                </ul>
                <p className="mt-6">Both branches are concatenated into a unified feature map that goes through a final convolutional block to yield rich spatial representations that capture both local and broad contextual features.</p>
                <p className="mt-4 text-sm text-gray-600"><strong>Spatial Context:</strong> The 171 input channels are created by the <code>add_surrounding_position</code> function, which expands each pixel's 19 features to include information from its 8 neighboring pixels (center + 8 neighbors = 9 positions × 19 features = 171 channels). This provides crucial spatial context for wildfire spread prediction.</p>
            </div>

            <div className="flex flex-col items-start">
              <ZoomIn className="w-6 h-6 text-blue-700 mb-2" />
              <h4 className="text-lg font-semibold text-gray-800">
                Conv2DBlock
              </h4>
              <p>Key component within CNNModel:</p>
                <ul className="list-disc list-inside pl-5 space-y-1 text-left">
                  <p><strong>Conv2DBlock:</strong></p>
                  <ul className="list-disc list-inside pl-8 space-y-1">
                    <li>Uses depthwise convolutions with causal padding (CausalDWConv2D), capturing spatial patterns.</li>
                    <li>Implements attention mechanisms (ECA) to dynamically weigh channel importance.</li>
                    <li>Applies normalisation, spatial attention, dropout, and non-linear activations (SILU).</li>
                  </ul>
                </ul>
            </div>

            <div className="flex flex-col items-start">
              <ArrowBigRightDash className="w-6 h-6 text-green-700 mb-2" />
              <h4 className="text-lg font-semibold text-gray-800">
                Next-Day Prediction (NextFramePredictor)
              </h4>
              <p><strong>Input:</strong> Feature maps from CNNModel (32×32×128)</p>
              <p>Receives feature maps from CNNModel:</p>
                <ul className="list-disc list-inside pl-5 space-y-1 text-left">
                  <p><strong>Conv2DBlock:</strong></p>
                  <ul className="list-disc list-inside pl-8 space-y-1">
                    <li>Uses 3-layer CNN (128 → 64 → 32 → 1) with ReLU activations, concluding with a Sigmoid activation.</li>
                    <li>Predicts the probability of wildfire presence at each spatial location with the output shape: height × width × 1</li>
                  </ul>
                </ul>
                <p className="mt-6">Our best model (v3) achieved an F1 score of 0.425, IoU of 0.270, precision of 0.312, recall of 0.669, and an inference speed of 51.0 ms.</p>
                <p className="mt-2 text-sm text-gray-600"><strong>Note:</strong> Predictions are upscaled back to 64×64 for visualization and metrics calculation to ensure proper alignment with the original ground truth data.</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="mt-8 mb-8">
        <CardHeader>
          <CardTitle>Custom Loss Function</CardTitle>
          <CardDescription>
            Addressing class imbalance with Weighted Binary Cross-Entropy (WBCE) and Dice Loss
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800 mb-3">Challenge: Class Imbalance</h4>
              <div className="space-y-2 text-sm text-gray-600">
                <p>• <strong>Fire pixels</strong>: Only 1.34% of the dataset (severely underrepresented)</p>
                <p>• <strong>No-fire pixels</strong>: 97.81% of the dataset (majority class)</p>
                <p>• <strong>Problem</strong>: Standard BCE loss would bias the model toward predicting "no-fire"</p>
              </div>
              
              <h4 className="text-lg font-semibold text-gray-800 mb-3 mt-6">Solution: WBCE + Dice Loss</h4>
              <div className="space-y-2 text-sm text-gray-600">
                <p>• <strong>Weighted Binary Cross-Entropy (WBCE)</strong>: Assigns higher weights to fire class</p>
                <p>• <strong>Fire class weight (w₁)</strong>: 10 (10x higher importance)</p>
                <p>• <strong>No-fire class weight (w₀)</strong>: 1 (baseline importance)</p>
                <p>• <strong>Dice Loss</strong>: Improves segmentation accuracy and boundary precision</p>
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800 mb-3">Mathematical Formulation</h4>
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-sm font-medium text-gray-700 mb-2">WBCE Loss:</p>
                <p className="text-xs text-gray-600 font-mono mb-3">
                  WBCE = -1/N × Σ(wᵢ × [yᵢ × log(ŷᵢ) + (1-yᵢ) × log(1-ŷᵢ)])
                </p>
                
                <p className="text-sm font-medium text-gray-700 mb-2">Weight Assignment:</p>
                <p className="text-xs text-gray-600 font-mono mb-3">
                  wᵢ = {'{'}w₁ = 10 if yᵢ = 1 (fire), w₀ = 1 if yᵢ = 0 (no-fire){'}'}
                </p>
                
                <p className="text-sm font-medium text-gray-700 mb-2">Combined Loss:</p>
                <p className="text-xs text-gray-600 font-mono">
                  Total Loss = WBCE + 2 × Dice Loss
                </p>
              </div>
              
              <div className="mt-4 space-y-2 text-sm text-gray-600">
                <p>• <strong>Factor of 2</strong>: Emphasizes Dice Loss contribution for precise boundaries</p>
                <p>• <strong>Masking</strong>: Excludes invalid pixels (marked as -1) from loss computation</p>
                <p>• <strong>Result</strong>: Balanced training that prioritizes fire detection while maintaining accuracy</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

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
  );
}