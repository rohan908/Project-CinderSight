import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'CinderSight - Canadian Fire Prediction',
  description: 'Advanced wildfire prediction using Canadian Fire Database and AI',
  keywords: 'wildfire, prediction, Canada, fire database, AI, machine learning',
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
          {children}
        </div>
      </body>
    </html>
  )
} 