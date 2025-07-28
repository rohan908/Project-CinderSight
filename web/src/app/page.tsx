'use client'

import { useState, useEffect } from 'react'
import { 
  Flame, 
  MapPin,  
  AlertTriangle, 
  TrendingUp, 
  Globe,
  Loader2,
  BarChart3,
  Image as ImageIcon,
  Settings
} from 'lucide-react'
import Image from 'next/image'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'

interface SampleData {
  sample_idx: number
  metrics: {
    f1: number
    iou: number
    precision: number
    recall: number
    tp: number
    fp: number
    fn: number
    tn: number
  }
  images: {
    previous_fire: string
    ground_truth: string
    prediction_probability: string
    prediction_binary: string
    comparison: string
    metrics_performance_chart: string
    metrics_confusion_matrix: string
    feature_images: Record<string, string>
  }
}

const FEATURES = [
  { id: 'elevation', name: 'Elevation', description: 'Terrain elevation in meters', category: 'Terrain' },
  { id: 'temperature', name: 'Temperature', description: 'Current temperature in °C', category: 'Weather' },
  { id: 'humidity', name: 'Humidity', description: 'Relative humidity percentage', category: 'Weather' },
  { id: 'wind_speed', name: 'Wind Speed', description: 'Wind speed in m/s', category: 'Weather' },
  { id: 'wind_direction', name: 'Wind Direction', description: 'Wind direction in degrees', category: 'Weather' },
  { id: 'precipitation', name: 'Precipitation', description: 'Precipitation in mm', category: 'Weather' },
  { id: 'pressure', name: 'Pressure', description: 'Atmospheric pressure in hPa', category: 'Weather' },
  { id: 'solar_radiation', name: 'Solar Radiation', description: 'Solar radiation in W/m²', category: 'Weather' },
  { id: 'visibility', name: 'Visibility', description: 'Visibility in km', category: 'Weather' },
  { id: 'slope', name: 'Slope', description: 'Terrain slope in degrees', category: 'Terrain' },
  { id: 'aspect', name: 'Aspect', description: 'Terrain aspect in degrees', category: 'Terrain' },
  { id: 'ndvi', name: 'NDVI', description: 'Normalized Difference Vegetation Index', category: 'Vegetation' },
  { id: 'land_cover', name: 'Land Cover', description: 'Land cover type', category: 'Vegetation' },
  { id: 'population', name: 'Population', description: 'Population density in people/km²', category: 'Human' },
]

export default function Home() {
  const [sampleData, setSampleData] = useState<SampleData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedSample, setSelectedSample] = useState<number | null>(null)
  const [selectedFeature, setSelectedFeature] = useState('elevation')
  const [availableSamples, setAvailableSamples] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [isMobile, setIsMobile] = useState(false)
  const [featureLoading, setFeatureLoading] = useState(false)

  // Check if mobile
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768)
    }
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  // Load sample count on mount
  useEffect(() => {
    fetchSampleCount()
  }, [])

  // Load random sample on mount
  useEffect(() => {
    if (availableSamples > 0) {
      loadRandomSample()
    }
  }, [availableSamples])

  const fetchSampleCount = async () => {
    try {
      console.log('Fetching sample count...')
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/samples/count`)
      console.log('Sample count response status:', response.status)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      console.log('Sample count data:', data)
      setAvailableSamples(data.total_samples)
    } catch (error) {
      console.error('Error fetching sample count:', error)
      setError(`Failed to load sample count: ${error}`)
    }
  }

  const loadRandomSample = async () => {
    setLoading(true)
    setError(null)
    
    try {
      // Generate random sample index
      const randomIndex = Math.floor(Math.random() * availableSamples)
      setSelectedSample(randomIndex)
      
      // Generate visualizations for this sample
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/visualization/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sample_idx: randomIndex,
          save_images: false,
          overwrite_existing: true,
          include_features: true,
          include_fire_progression: true,
          include_metrics_dashboard: true,
          include_documentation: true
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to generate visualizations')
      }

      const result = await response.json()
      const taskId = result.task_id

      // Poll for completion
      await pollForCompletion(taskId)
      
    } catch (error) {
      console.error('Error loading sample:', error)
      setError('Failed to load sample data')
    } finally {
      setLoading(false)
    }
  }

  const pollForCompletion = async (taskId: string) => {
    const maxAttempts = 60 // 5 minutes with 5-second intervals
    let attempts = 0
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

    while (attempts < maxAttempts) {
      try {
        const response = await fetch(`${apiUrl}/visualization/status/${taskId}`)
        const status = await response.json()

        if (status.status === 'completed') {
          // Use metrics from status response
          const metrics = status.metrics
          
          // Download feature images
          const featuresResponse = await fetch(`${apiUrl}/visualization/download/${taskId}/features`)
          const featuresBlob = await featuresResponse.blob()
          
          const imageUrls = {
            previous_fire: `${apiUrl}/visualization/image/${taskId}/fire_previous_fire.png`,
            ground_truth: `${apiUrl}/visualization/image/${taskId}/fire_ground_truth.png`,
            prediction_probability: `${apiUrl}/visualization/image/${taskId}/fire_prediction_probability.png`,
            prediction_binary: `${apiUrl}/visualization/image/${taskId}/fire_prediction_binary.png`,
            comparison: `${apiUrl}/visualization/image/${taskId}/fire_comparison_overlay.png`,
            metrics_performance_chart: `${apiUrl}/visualization/image/${taskId}/metrics_performance_chart.png`,
            metrics_confusion_matrix: `${apiUrl}/visualization/image/${taskId}/metrics_confusion_matrix.png`,
            feature_images: {
              elevation: `${apiUrl}/visualization/image/${taskId}/feature_12_elevation.png`,
              temperature: `${apiUrl}/visualization/image/${taskId}/feature_03_tmmx.png`,
              humidity: `${apiUrl}/visualization/image/${taskId}/feature_02_sph.png`,
              wind_speed: `${apiUrl}/visualization/image/${taskId}/feature_00_vs.png`,
              wind_direction: `${apiUrl}/visualization/image/${taskId}/feature_05_th.png`,
              precipitation: `${apiUrl}/visualization/image/${taskId}/feature_01_pr.png`,
              pressure: `${apiUrl}/visualization/image/${taskId}/feature_07_pdsi.png`,
              solar_radiation: `${apiUrl}/visualization/image/${taskId}/feature_06_erc.png`,
              visibility: `${apiUrl}/visualization/image/${taskId}/feature_06_erc.png`,
              slope: `${apiUrl}/visualization/image/${taskId}/feature_14_slope.png`,
              aspect: `${apiUrl}/visualization/image/${taskId}/feature_13_aspect.png`,
              ndvi: `${apiUrl}/visualization/image/${taskId}/feature_15_ndvi.png`,
              land_cover: `${apiUrl}/visualization/image/${taskId}/feature_16_evi.png`,
              population: `${apiUrl}/visualization/image/${taskId}/feature_17_population.png`,
            }
          }
           
           console.log('📊 Metrics received:', metrics)
           console.log('🖼️ Image URLs created:', imageUrls)

          setSampleData({
            sample_idx: selectedSample!,
            metrics: metrics,
            images: imageUrls
          })
          return
        } else if (status.status === 'failed') {
          throw new Error(status.error_message || 'Task failed')
        }

        await new Promise(resolve => setTimeout(resolve, 5000)) // Wait 5 seconds
        attempts++
      } catch (error) {
        console.error('Error polling task status:', error)
        throw error
      }
    }
    
    throw new Error('Task timed out')
  }

  const loadSpecificSample = async (sampleIdx: number) => {
    setLoading(true)
    setError(null)
    
    try {
      setSelectedSample(sampleIdx)
      
      // Generate visualizations for this sample
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/visualization/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sample_idx: sampleIdx,
          save_images: false,
          overwrite_existing: true,
          include_features: true,
          include_fire_progression: true,
          include_metrics_dashboard: true,
          include_documentation: true
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to generate visualizations')
      }

      const result = await response.json()
      const taskId = result.task_id

      // Poll for completion
      await pollForCompletion(taskId)
      
    } catch (error) {
      console.error('Error loading sample:', error)
      setError('Failed to load sample data')
    } finally {
      setLoading(false)
    }
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8">
        <Card className="max-w-md mx-auto">
          <CardHeader>
            <CardTitle className="text-red-600 flex items-center">
              <AlertTriangle className="w-5 h-5 mr-2" />
              Error
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-600 mb-4">{error}</p>
            <Button onClick={loadRandomSample} className="w-full">
              Try Again
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="flex items-center justify-center mb-4">
          <Flame className="w-12 h-12 text-orange-500 mr-3" />
          <h1 className="text-4xl font-bold text-gray-900">CinderSight</h1>
        </div>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          Deep Learning Wildfire Spread Prediction Model
        </p>
      </div>

      {/* Sample Selection */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center">
            <MapPin className="w-6 h-6 mr-2 text-blue-600" />
            Sample Selection
          </CardTitle>
          <CardDescription>
            Choose a sample to analyze wildfire spread predictions
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <label className="text-sm font-medium mb-2 block">Current Sample</label>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-lg px-3 py-1">
                {selectedSample !== null ? selectedSample : 'Loading...'}
              </Badge>
              <span className="text-sm text-gray-500">
                of {availableSamples} available
              </span>
            </div>
          </div>
                    <div className="flex gap-2 items-center">
            <Button 
              onClick={loadRandomSample} 
              disabled={loading}
              className="flex-1 sm:flex-none"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Globe className="w-4 h-4 mr-2" />
              )}
              Random Sample
            </Button>
            {!isMobile && (
              <>
                <input
                  type="number"
                  min="0"
                  max={availableSamples - 1}
                  placeholder="Sample #"
                  className="w-24 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 h-9"
                  style={{ height: '36px' }}
                  value={selectedSample?.toString() || ''}
                  onChange={(e) => {
                    const value = parseInt(e.target.value)
                    if (!isNaN(value) && value >= 0 && value < availableSamples) {
                      setSelectedSample(value)
                    }
                  }}
                  disabled={loading}
                />
                <Button 
                  onClick={() => selectedSample !== null && loadSpecificSample(selectedSample)}
                  disabled={loading || selectedSample === null}
                  size="sm"
                >
                  Load
                </Button>
              </>
            )}
          </div>
        </CardContent>
      </Card>

      {loading ? (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Loading Sample Data...</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center space-x-4">
                <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
                <div>
                  <p className="text-sm font-medium">Generating visualizations</p>
                  <p className="text-sm text-gray-500">This may take a few moments...</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Loading skeletons */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {[1, 2, 3].map((i) => (
              <Card key={i}>
                <CardHeader>
                  <Skeleton className="h-6 w-32" />
                </CardHeader>
                <CardContent>
                  <Skeleton className="h-48 w-full" />
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      ) : sampleData ? (
        <div className="space-y-6">
          {/* Metrics Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-600">F1 Score</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-bold">
                    {(sampleData.metrics.f1 * 100).toFixed(1)}%
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-600">IoU</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-bold">
                    {(sampleData.metrics.iou * 100).toFixed(1)}%
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-600">Precision</CardTitle>
              </CardHeader>
              <CardContent>
                <span className="text-2xl font-bold">
                  {(sampleData.metrics.precision * 100).toFixed(1)}%
                </span>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-600">Recall</CardTitle>
              </CardHeader>
              <CardContent>
                <span className="text-2xl font-bold">
                  {(sampleData.metrics.recall * 100).toFixed(1)}%
                </span>
            </CardContent>
          </Card>
        </div>

          {/* Always Visible Fire Masks */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Flame className="w-6 h-6 mr-2 text-red-600" />
                Fire Spread Analysis
              </CardTitle>
              <CardDescription>
                Previous fire, prediction, and comparison visualizations
              </CardDescription>
            </CardHeader>
            <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                 <div className="space-y-2">
                   <h4 className="font-medium text-sm">Ground Truth</h4>
                   <div className="aspect-[4/3] relative rounded-lg border overflow-hidden bg-gray-50">
                     <Image 
                       src={sampleData.images.ground_truth} 
                       alt="Ground Truth Fire Mask"
                       fill
                       className="object-contain"
                       onLoad={() => console.log('✅ Ground truth image loaded successfully')}
                       onError={() => console.error('❌ Ground truth image failed to load')}
                     />
                   </div>
                 </div>
                 <div className="space-y-2">
                   <h4 className="font-medium text-sm">Prediction (Probability)</h4>
                   <div className="aspect-[4/3] relative rounded-lg border overflow-hidden bg-gray-50">
                     <Image 
                       src={sampleData.images.prediction_probability} 
                       alt="Prediction Probability"
                       fill
                       className="object-contain"
                       onLoad={() => console.log('✅ Prediction probability image loaded successfully')}
                       onError={() => console.error('❌ Prediction probability image failed to load')}
                     />
                   </div>
                 </div>
                 <div className="space-y-2">
                   <h4 className="font-medium text-sm">IoU Comparison</h4>
                   <div className="aspect-[4/3] relative rounded-lg border overflow-hidden bg-gray-50">
                     <Image 
                       src={sampleData.images.comparison} 
                       alt="IoU Comparison"
                       fill
                       className="object-contain"
                       onLoad={() => console.log('✅ IoU comparison image loaded successfully')}
                       onError={() => console.error('❌ IoU comparison image failed to load')}
                     />
                   </div>
                 </div>
              </div>
            </CardContent>
          </Card>

          {/* Feature Selection and Display */}
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Feature Sidebar */}
            <Card className="lg:col-span-1">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Settings className="w-5 h-5 mr-2" />
                  Input Features
                </CardTitle>
                <CardDescription>
                  Select a feature to visualize
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isMobile ? (
                  <Select value={selectedFeature} onValueChange={(value: string) => {
                    setFeatureLoading(true)
                    setSelectedFeature(value)
                    // Simulate loading delay for feature switching
                    setTimeout(() => setFeatureLoading(false), 500)
                  }}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {FEATURES.map((feature) => (
                        <SelectItem key={feature.id} value={feature.id}>
                          {feature.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <div className="space-y-2">
                    {FEATURES.map((feature) => (
                      <button
                        key={feature.id}
                        onClick={() => {
                          setFeatureLoading(true)
                          setSelectedFeature(feature.id)
                          // Simulate loading delay for feature switching
                          setTimeout(() => setFeatureLoading(false), 500)
                        }}
                        className={`w-full text-left p-2 rounded-lg transition-colors ${
                          selectedFeature === feature.id
                            ? 'bg-blue-100 text-blue-900'
                            : 'hover:bg-gray-100'
                        }`}
                      >
                        <div className="font-medium text-sm">{feature.name}</div>
                        <div className="text-xs text-gray-500">{feature.description}</div>
                        <Badge variant="outline" className="text-xs mt-1">
                          {feature.category}
                  </Badge>
                      </button>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Feature Visualization */}
            <Card className="lg:col-span-3">
        <CardHeader>
                <CardTitle className="flex items-center">
                  <BarChart3 className="w-5 h-5 mr-2" />
                  {FEATURES.find(f => f.id === selectedFeature)?.name} Visualization
                </CardTitle>
                <CardDescription>
                  {FEATURES.find(f => f.id === selectedFeature)?.description}
                </CardDescription>
        </CardHeader>
                <CardContent>
                 <div className="aspect-[4/3] relative rounded-lg border overflow-hidden bg-gray-50">
                   {featureLoading ? (
                     <div className="w-full h-full flex items-center justify-center bg-gray-100">
                       <div className="flex items-center space-x-2">
                         <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
                         <span className="text-sm text-gray-600">Loading feature...</span>
                       </div>
                     </div>
                   ) : (
                     <Image 
                       src={sampleData.images.feature_images[selectedFeature]} 
                       alt={`${FEATURES.find(f => f.id === selectedFeature)?.name} Visualization`}
                       fill
                       className="object-contain"
                       onLoad={() => console.log(`✅ ${selectedFeature} image loaded successfully`)}
                       onError={() => console.error(`❌ ${selectedFeature} image failed to load`)}
                     />
                   )}
              </div>
               </CardContent>
            </Card>
            </div>

                    {/* Metrics Dashboard */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <TrendingUp className="w-5 h-5 mr-2 text-green-600" />
                  Performance Metrics
                </CardTitle>
                <CardDescription>
                  Precision, Recall, F1 Score, and IoU
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative w-full aspect-[4/3] rounded-lg border overflow-hidden bg-gray-50">
                  <Image 
                    src={sampleData.images.metrics_performance_chart} 
                    alt="Performance Metrics Chart"
                    fill
                    className="object-contain"
                    onLoad={() => console.log('✅ Performance metrics chart loaded successfully')}
                    onError={() => console.error('❌ Performance metrics chart failed to load')}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <BarChart3 className="w-5 h-5 mr-2 text-blue-600" />
                  Confusion Matrix
                </CardTitle>
                <CardDescription>
                  True Positives, False Positives, True Negatives, False Negatives
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative w-full aspect-[4/3] rounded-lg border overflow-hidden bg-gray-50">
                  <Image 
                    src={sampleData.images.metrics_confusion_matrix} 
                    alt="Confusion Matrix"
                    fill
                    className="object-contain"
                    onLoad={() => console.log('✅ Confusion matrix loaded successfully')}
                    onError={() => console.error('❌ Confusion matrix failed to load')}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      ) : null}
    </div>
  )
} 