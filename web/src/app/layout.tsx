// web/src/app/layout.tsx
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import NavBar from '@/components/NavBar'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'CinderSight - Wildfire Fire Prediction',
  description: 'Advanced wildfire prediction using Wildfire Fire Database and AI',
  keywords: 'wildfire, prediction, fire database, AI, machine learning',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gradient-to-br from-orange-50 to-red-50">
          {/* Navbar */}
          <header className="bg-white bg-opacity-80 backdrop-blur-sm">
            <NavBar />
          </header>

          {/* Page content */}
          <main>{children}</main>
        </div>
      </body>
    </html>
  )
}
