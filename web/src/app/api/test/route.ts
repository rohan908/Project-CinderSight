import { NextRequest, NextResponse } from 'next/server'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    console.log(`🧪 Testing API connection to: ${API_BASE_URL}/health`)
    
    const response = await fetch(`${API_BASE_URL}/health`, {
      headers: {
        'Accept': 'application/json',
      },
    })
    
    console.log(`📊 Test response status: ${response.status}`)
    
    if (!response.ok) {
      return NextResponse.json(
        { error: 'Backend not reachable', status: response.status },
        { status: 500 }
      )
    }
    
    const data = await response.json()
    return NextResponse.json({
      message: 'API connection successful',
      backend: data,
      apiUrl: API_BASE_URL
    })
  } catch (error) {
    console.error('❌ API test error:', error)
    return NextResponse.json(
      { error: 'API connection failed', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
} 