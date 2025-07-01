'use client'

import { useState } from 'react'
import { Flame, MapPin, Calendar, AlertTriangle, TrendingUp, Globe } from 'lucide-react'
import dynamic from 'next/dynamic'
import { format } from 'date-fns'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

// Dynamically import the map component to avoid SSR issues
const FireMap = dynamic(() => import('@/components/FireMap'), { ssr: false })

interface PredictionResult {
  risk_level: 'low' | 'medium' | 'high' | 'extreme'
  probability: number
  spread_direction: string
  estimated_area: number
  confidence: number
}

export default function Home() {
  const [selectedLocation, setSelectedLocation] = useState<[number, number] | null>(null)
  const [selectedDate, setSelectedDate] = useState<string>(format(new Date(), 'yyyy-MM-dd'))
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)

  const handleLocationSelect = (lat: number, lng: number) => {
    setSelectedLocation([lat, lng])
  }

  const handlePredict = async () => {
    if (!selectedLocation) return

    setLoading(true)
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ignition_point: {
            latitude: selectedLocation[0],
            longitude: selectedLocation[1]
          },
          date: selectedDate
        }),
      })

      if (response.ok) {
        const result = await response.json()
        setPrediction(result.prediction)
      } else {
        console.error('Prediction failed')
      }
    } catch (error) {
      console.error('Error making prediction:', error)
    } finally {
      setLoading(false)
    }
  }

  const getRiskBadgeVariant = (risk: string) => {
    switch (risk) {
      case 'low': return 'default'
      case 'medium': return 'secondary'
      case 'high': return 'destructive'
      case 'extreme': return 'destructive'
      default: return 'default'
    }
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
          Advanced wildfire prediction powered by the Canadian Fire Database and AI
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Map Section */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Globe className="w-6 h-6 mr-2 text-blue-600" />
                Select Location
              </CardTitle>
              <CardDescription>
                Click on the map to select a location in Canada for fire prediction
              </CardDescription>
            </CardHeader>
            <CardContent>
              <FireMap 
                onLocationSelect={handleLocationSelect}
                selectedLocation={selectedLocation}
              />
              {selectedLocation && (
                <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm text-blue-800">
                    Selected: {selectedLocation[0].toFixed(4)}, {selectedLocation[1].toFixed(4)}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Prediction Form */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <TrendingUp className="w-6 h-6 mr-2 text-green-600" />
                Fire Prediction
              </CardTitle>
              <CardDescription>
                Choose a date and get fire risk predictions for the selected location
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="date" className="flex items-center">
                  <Calendar className="w-4 h-4 mr-2" />
                  Prediction Date
                </Label>
                <Input
                  id="date"
                  type="date"
                  value={selectedDate}
                  onChange={(e) => setSelectedDate(e.target.value)}
                />
              </div>

              <Button
                onClick={handlePredict}
                disabled={!selectedLocation || loading}
                className="w-full"
                size="lg"
              >
                {loading ? (
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                ) : (
                  <>
                    <Flame className="w-5 h-5 mr-2" />
                    Predict Fire Risk
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Results */}
          {prediction && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <AlertTriangle className="w-5 h-5 mr-2 text-red-600" />
                  Prediction Results
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <h4 className="font-semibold">Risk Level</h4>
                    <p className="text-sm text-gray-600">Probability: {(prediction.probability * 100).toFixed(1)}%</p>
                  </div>
                  <Badge variant={getRiskBadgeVariant(prediction.risk_level)}>
                    {prediction.risk_level.toUpperCase()}
                  </Badge>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-600">Spread Direction</p>
                    <p className="font-semibold">{prediction.spread_direction}</p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-600">Estimated Area</p>
                    <p className="font-semibold">{prediction.estimated_area.toFixed(1)} km²</p>
                  </div>
                </div>

                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm text-blue-600">Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Features Section */}
      <Card className="mt-12">
        <CardHeader>
          <CardTitle className="text-center text-3xl">Why CinderSight?</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Flame className="w-8 h-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Real-time Data</h3>
              <p className="text-gray-600">Powered by the comprehensive Canadian Fire Database with live updates</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <TrendingUp className="w-8 h-8 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">AI-Powered</h3>
              <p className="text-gray-600">Advanced machine learning models for accurate fire spread predictions</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <MapPin className="w-8 h-8 text-orange-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Interactive Maps</h3>
              <p className="text-gray-600">Visualize fire risks and spread patterns on detailed maps</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 