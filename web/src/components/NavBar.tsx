// web/src/components/NavBar.tsx
'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

interface NavItem {
  href: string
  label: string
  external?: boolean
}

const navItems: NavItem[] = [
  { href: '/',       label: 'Home'   },
  { href: '/about',  label: 'About'  },
  { href: '/data',   label: 'Data'   },
  { href: '/model',  label: 'Model'  },
  {
    href: 'https://github.com/rohan908/Project-CinderSight',
    label: 'GitHub',
    external: true,
  },
]

export default function NavBar() {
  const pathname = usePathname() || '/'

  return (
    <div className="container mx-auto px-4 py-4 flex space-x-6">
      {navItems.map((item) => {
        const isInternal = !item.external
        const isActive =
          isInternal &&
          (item.href === '/'
            ? pathname === '/'
            : pathname.startsWith(item.href))

        const baseClasses =
          'font-medium ' +
          (isActive
            ? 'text-orange-600 border-b-2 border-orange-600'
            : 'text-gray-800 hover:text-orange-600')

        return item.external ? (
          <a
            key={item.href}
            href={item.href}
            target="_blank"
            rel="noopener noreferrer"
            className={baseClasses}
          >
            {item.label}
          </a>
        ) : (
          <Link key={item.href} href={item.href} className={baseClasses}>
            {item.label}
          </Link>
        )
      })}
    </div>
  )
}
