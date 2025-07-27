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


export default function AboutPage() {
  return (
    <div className="container mx-auto px-4 py-8 space-y-6">
      <h1 className="text-4xl font-bold text-gray-900">About CinderSight 🔥</h1>
      <p className="text-xl text-gray-600">
        CinderSight is an advanced wildfire prediction tool using satellite and environmental data to
        forecast fire spread across Canada. Our mission is to empower responders and communities
        with real-time risk assessments.
      </p>
      <p className="text-xl text-gray-600">
        Under the hood, we train a custom deep-learning model on the Canadian Fire Spread Dataset,
        combining terrain, weather, vegetation, and human-factors covariates.
      </p>
    </div>
  )
}
