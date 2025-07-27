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

export default function DataPage() {
  return (
    <div className="container mx-auto px-4 py-8 space-y-6">
      <h1 className="text-4xl font-bold text-gray-900">Data Used 📊</h1>
      <p className="text-xl text-gray-600">
        We leverage the <strong>Canadian Fire Spread Dataset (CFSDS)</strong>, which provides daily
        fire progression maps (180 m pixels) for all ≥ 1 000 ha fires in Canada (2002–2021), with
        50 environmental covariates per pixel.
      </p>
      <p className="text-xl text-gray-600">
        🔗{' '}
        <a
          className="text-blue-600 underline"
          href="https://osf.io/f48ry/"
          target="_blank"
          rel="noopener noreferrer"
        >
          Explore the dataset on OSF
        </a>
      </p>
    </div>
  )
}
