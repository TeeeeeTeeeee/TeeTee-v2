"use client";

import { IconArrowLeft, IconArrowRight } from "@tabler/icons-react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { useEffect, useMemo, useState } from "react";
import { cn } from "@/lib/utils";

// Types
export type Testimonial = {
  quote: string;
  name: string;
  designation: string; // position in the company
  src: string;
};

/**
 * AnimatedTestimonials
 *
 * Sizing rules (Golden Ratio φ ≈ 1.618):
 * - Heading ("What other developers says"): text-2xl (1.5rem) - STAYS THE SAME
 * - Name: 2.427rem (1.5rem × φ) - upsized by golden ratio from heading
 * - Description (quote): name / φ = 2.427rem / φ = 1.5rem (golden ratio smaller than name)
 * - Position: description / φ = 1.5rem / φ = 0.927rem (golden ratio smaller than description)
 * - Picture: name * φ² = 2.427rem * φ² (2 times golden ratio bigger than name)
 */
export const AnimatedTestimonials = ({
  testimonials,
  autoplay = false,
  className,
  nameSizeRem = 2.427, // 1.5rem × φ - upsized by golden ratio from heading
}: {
  testimonials: Testimonial[];
  autoplay?: boolean;
  className?: string;
  /** Base font size in rem for the name (φ times larger than heading). */
  nameSizeRem?: number;
}) => {
  const [active, setActive] = useState(0);

  const PHI = 1.61803398875;

  // Derived sizes from name size
  const { quoteSizeRem, designationSizeRem, imageSizePx } = useMemo(() => {
    const quote = nameSizeRem / PHI; // 1.5rem - golden ratio smaller than name
    const designation = quote / PHI; // 0.927rem - golden ratio smaller than description
    // Image: 2 times golden ratio (φ²) bigger than name
    const imagePx = Math.max(280, Math.round(nameSizeRem * PHI * PHI * 16)); // name * φ²
    return { quoteSizeRem: quote, designationSizeRem: designation, imageSizePx: imagePx };
  }, [nameSizeRem]);

  const handleNext = () => setActive((prev) => (prev + 1) % testimonials.length);
  const handlePrev = () => setActive((prev) => (prev - 1 + testimonials.length) % testimonials.length);
  const isActive = (index: number) => index === active;

  useEffect(() => {
    if (!autoplay) return;
    const interval = setInterval(handleNext, 5000);
    return () => clearInterval(interval);
  }, [autoplay]);

  const randomRotateY = () => Math.floor(Math.random() * 21) - 10;

  if (!testimonials?.length) return null;

  return (
    <div className={cn("max-w-7xl mx-auto px-4 md:px-8 lg:px-12 py-20", className)}>
      {/* Heading: text-2xl (1.5rem) - STAYS THE SAME (not upsized) */}
      <h3 className="text-2xl font-medium text-gray-500 uppercase tracking-wider text-center mb-16">
        What other developers says
      </h3>

      <div className="relative grid grid-cols-1 md:grid-cols-2 gap-20 items-start">
        {/* Image stack - 2 times golden ratio (φ²) bigger than name */}
        <div className="flex items-center justify-center">
          <div className="relative w-full max-w-full" style={{ width: imageSizePx, height: imageSizePx }}>
            <AnimatePresence>
              {testimonials.map((testimonial, index) => (
                <motion.div
                  key={testimonial.src}
                  initial={{ opacity: 0, scale: 0.9, rotate: randomRotateY() }}
                  animate={{
                    opacity: isActive(index) ? 1 : 0.7,
                    scale: isActive(index) ? 1 : 0.95,
                    rotate: isActive(index) ? 0 : randomRotateY(),
                    zIndex: isActive(index) ? 999 : testimonials.length + 2 - index,
                    y: isActive(index) ? [0, -20, 0] : 0,
                  }}
                  exit={{ opacity: 0, scale: 0.9, rotate: randomRotateY() }}
                  transition={{ duration: 0.4, ease: "easeInOut" }}
                  className="absolute inset-0 origin-bottom"
                >
                  <Image
                    src={testimonial.src}
                    alt={testimonial.name}
                    width={imageSizePx}
                    height={imageSizePx}
                    draggable={false}
                    className="h-full w-full rounded-3xl object-cover object-center"
                  />
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* Copy */}
        <div className="flex justify-between flex-col py-4">
          <motion.div
            key={active}
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -20, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
          >
            {/* Name — 2.427rem (heading × φ) - upsized by golden ratio */}
            <h4 className="font-bold text-foreground leading-tight" style={{ fontSize: `${nameSizeRem}rem` }}>
              {testimonials[active].name}
            </h4>

            {/* Position — 0.927rem (description / φ) - golden ratio smaller than description */}
            <p className="text-muted-foreground mt-2" style={{ fontSize: `${designationSizeRem}rem` }}>
              {testimonials[active].designation}
            </p>

            {/* Description (Quote) — 1.5rem (name / φ) - golden ratio smaller than name */}
            <motion.p
              className="text-muted-foreground mt-8 leading-relaxed"
              style={{ fontSize: `${quoteSizeRem}rem` }}
            >
              {testimonials[active].quote.split(" ").map((word, index) => (
                <motion.span
                  key={index}
                  initial={{ filter: "blur(10px)", opacity: 0, y: 5 }}
                  animate={{ filter: "blur(0px)", opacity: 1, y: 0 }}
                  transition={{ duration: 0.2, ease: "easeInOut", delay: 0.02 * index }}
                  className="inline-block"
                >
                  {word}&nbsp;
                </motion.span>
              ))}
            </motion.p>
          </motion.div>

          <div className="flex gap-4 pt-12 md:pt-8">
            <button
              onClick={handlePrev}
              className="h-12 w-12 rounded-full bg-secondary flex items-center justify-center group/button"
              aria-label="Previous testimonial"
            >
              <IconArrowLeft className="h-6 w-6 text-foreground group-hover/button:rotate-12 transition-transform duration-300" />
            </button>
            <button
              onClick={handleNext}
              className="h-12 w-12 rounded-full bg-secondary flex items-center justify-center group/button"
              aria-label="Next testimonial"
            >
              <IconArrowRight className="h-6 w-6 text-foreground group-hover/button:-rotate-12 transition-transform duration-300" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnimatedTestimonials;
