import { NextRequest, NextResponse } from 'next/server'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const path = params.path.join('/')
  
  try {
    console.log(`🖼️ Fetching image: ${API_BASE_URL}/visualization/image/${path}`)
    
    const response = await fetch(`${API_BASE_URL}/visualization/image/${path}`, {
      headers: {
        'Accept': 'image/*',
      },
    })
    
    console.log(`📊 Image response status: ${response.status}`)
    
    if (!response.ok) {
      console.error(`❌ Image not found: ${response.status} ${response.statusText}`)
      return NextResponse.json(
        { error: 'Image not found' },
        { status: 404 }
      )
    }
    
    // Get the image data as a buffer
    const imageBuffer = await response.arrayBuffer()
    console.log(`✅ Image buffer size: ${imageBuffer.byteLength} bytes`)
    
    // Get the content type from the response
    const contentType = response.headers.get('content-type') || 'image/png'
    console.log(`📋 Content type: ${contentType}`)
    
    // Return the image with proper headers
    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=3600', // Cache for 1 hour
        'Access-Control-Allow-Origin': '*',
      },
    })
  } catch (error) {
    console.error('❌ Image proxy error:', error)
    return NextResponse.json(
      { error: 'Failed to load image' },
      { status: 500 }
    )
  }
} 