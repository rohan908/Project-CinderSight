// web/src/app/github/page.tsx
import { Code, Terminal, Layers } from 'lucide-react'
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from '@/components/ui/card'

export default function GitHubPage() {
  return (
    <div className="container mx-auto px-4 py-8 space-y-12">
      {/* Page Header */}
      <div className="text-center">
        <div className="flex items-center justify-center space-x-4 mb-4">
          <Code className="w-10 h-10 text-gray-800" />
          <h1 className="text-4xl font-bold text-gray-900">Code & Setup 🛠️</h1>
        </div>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-2">
          Explore our entire codebase on GitHub.
        </p>
        <p className="text-lg text-gray-600">
          🔗{' '}
          <a
            href="https://github.com/rohan908/Project-CinderSight"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 underline"
          >
            Project-CinderSight
          </a>
        </p>
      </div>

      <hr className="border-t border-gray-200" />

      {/* Quickstart */}
      <Card className="bg-gray-50 shadow-lg">
        <CardHeader>
          <CardTitle className="text-center text-3xl">Quickstart 🏃‍♂️</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Clone */}
            <div className="text-center space-y-2">
              <div className="w-16 h-16 bg-orange-100 rounded-full flex items-center justify-center mx-auto">
                <Terminal className="w-8 h-8 text-orange-600" />
              </div>
              <h3 className="text-xl font-semibold">Clone</h3>
              <p className="text-sm font-mono whitespace-pre-wrap break-words text-gray-700">
                git clone https://github.com/rohan908/Project-CinderSight.git
              </p>
            </div>

            {/* Frontend */}
            <div className="text-center space-y-2">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto">
                <Layers className="w-8 h-8 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold">Frontend</h3>
              <p className="text-sm font-mono whitespace-pre-wrap break-words text-gray-700">
                cd web && npm install && npm run dev
              </p>
            </div>

            {/* Backend */}
            <div className="text-center space-y-2">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto">
                <Layers className="w-8 h-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold">Backend</h3>
              <p className="text-sm font-mono whitespace-pre-wrap break-words text-gray-700">
                cd api && pip install -r requirements.txt && uvicorn app.main:app --reload
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <hr className="border-t border-gray-200" />

      {/* Requirements */}
      <Card className="shadow-lg">
        <CardHeader>
          <CardTitle className="text-xl font-semibold">Requirements</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc list-inside space-y-1 text-base text-gray-600">
            <li>Node.js 18+</li>
            <li>Python 3.11+</li>
            <li>Docker &amp; Docker Compose (optional)</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
