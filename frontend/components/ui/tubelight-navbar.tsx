"use client"

import React, { useEffect, useState } from "react"
import { motion } from "framer-motion"
import Link from "next/link"
import { useRouter } from "next/router"
import { LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"

interface NavItem {
  name: string
  url: string
  icon: LucideIcon
}

interface NavBarProps {
  items: NavItem[]
  className?: string
}

export function NavBar({ items, className }: NavBarProps) {
  const router = useRouter()
  const [activeTab, setActiveTab] = useState(items[0].name)
  const [isMobile, setIsMobile] = useState(false)

  // Update active tab based on current route
  useEffect(() => {
    const currentItem = items.find(item => item.url === router.pathname)
    if (currentItem) {
      setActiveTab(currentItem.name)
    }
  }, [router.pathname, items])

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768)
    }

    handleResize()
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  return (
    <div className={cn("inline-block", className)}>
      <div className="flex items-center gap-2 bg-white py-1.5 px-1.5 rounded-full shadow-lg">
        {items.map((item) => {
          const Icon = item.icon
          const isActive = activeTab === item.name

          return (
            <Link
              key={item.name}
              href={item.url}
              onClick={() => setActiveTab(item.name)}
              className={cn(
                "relative cursor-pointer text-sm font-semibold px-5 py-2 rounded-full transition-all duration-200",
                "text-gray-700 hover:text-gray-900"
              )}
            >
              <span className="relative z-10">{item.name}</span>
              {isActive && (
                <motion.div
                  layoutId="tubelight"
                  className="absolute inset-0 bg-gray-100 rounded-full"
                  initial={false}
                  transition={{
                    type: "spring",
                    stiffness: 380,
                    damping: 30,
                  }}
                >
                  {/* Tubelight effect on top */}
                  <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-12 h-0.5 bg-black rounded-full" />
                </motion.div>
              )}
            </Link>
          )
        })}
      </div>
    </div>
  )
}

