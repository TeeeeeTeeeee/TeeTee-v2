"use client";

import { IconArrowLeft, IconArrowRight } from "@tabler/icons-react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

type Testimonial = {
  quote: string;
  name: string;
  designation: string;
  src: string;
};

export const AnimatedTestimonials = ({
  testimonials,
  autoplay = false,
  className,
}: {
  testimonials: Testimonial[];
  autoplay?: boolean;
  className?: string;
}) => {
  const [active, setActive] = useState(0);

  const handleNext = () => {
    setActive((prev) => (prev + 1) % testimonials.length);
  };

  const handlePrev = () => {
    setActive((prev) => (prev - 1 + testimonials.length) % testimonials.length);
  };

  const isActive = (index: number) => {
    return index === active;
  };

  useEffect(() => {
    if (autoplay) {
      const interval = setInterval(handleNext, 5000);
      return () => clearInterval(interval);
    }
  }, [autoplay]);

  const randomRotateY = () => {
    return Math.floor(Math.random() * 21) - 10;
  };

  const getCompanyLogo = (designation: string) => {
    const company = designation.split('@')[1]?.trim().toLowerCase();
    const logoMap: { [key: string]: string } = {
      'stripe': '/logos/stripe.png',
      'newrelic': '/logos/newrelic.png',
      'databricks': '/logos/databricks1.png',
      'zapier': '/logos/zapier.png',
      'datadog': '/logos/datadog.png',
      'openai': '/logos/openAI.png',
      'figma': '/logos/figma.png',
      'notion': '/logos/notion.png',
      'mongodb': '/logos/mongodb.png',
      'vercel': '/logos/vercel.png',
      'github': '/logos/github.png',
      'gitlab': '/logos/gitlab.png',
      'kubernetes': '/logos/kubernetes.png',
      'terraform': '/logos/terraform.png',
      'elastic': '/logos/elastic.png',
      'atlassian': '/logos/atlassian.png',
      'pagerduty': '/logos/pagerduty.png',
      'sentry': '/logos/sentry.png',
    };
    return logoMap[company] || '';
  };

  return (
    <div className={cn("max-w-7xl mx-auto px-4 md:px-8 lg:px-12 py-20", className)}>
      <h2 className="text-6xl font-medium text-gray-500 uppercase tracking-wider text-center mb-16">
        What other developers says
      </h2>
      <div className="relative grid grid-cols-1 md:grid-cols-2 gap-20">
        <div>
          <div className="relative h-[500px] w-full">
            <AnimatePresence>
              {testimonials.map((testimonial, index) => (
                <motion.div
                  key={testimonial.src}
                  initial={{
                    opacity: 0,
                    scale: 0.9,
                    z: -100,
                    rotate: randomRotateY(),
                  }}
                  animate={{
                    opacity: isActive(index) ? 1 : 0.7,
                    scale: isActive(index) ? 1 : 0.95,
                    z: isActive(index) ? 0 : -100,
                    rotate: isActive(index) ? 0 : randomRotateY(),
                    zIndex: isActive(index)
                      ? 999
                      : testimonials.length + 2 - index,
                    y: isActive(index) ? [0, -80, 0] : 0,
                  }}
                  exit={{
                    opacity: 0,
                    scale: 0.9,
                    z: 100,
                    rotate: randomRotateY(),
                  }}
                  transition={{
                    duration: 0.4,
                    ease: "easeInOut",
                  }}
                  className="absolute inset-0 origin-bottom"
                >
                  <Image
                    src={testimonial.src}
                    alt={testimonial.name}
                    width={500}
                    height={500}
                    draggable={false}
                    className="h-full w-full rounded-3xl object-cover object-center"
                  />
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
        <div className="flex justify-between flex-col py-4">
          <motion.div
            key={active}
            initial={{
              y: 20,
              opacity: 0,
            }}
            animate={{
              y: 0,
              opacity: 1,
            }}
            exit={{
              y: -20,
              opacity: 0,
            }}
            transition={{
              duration: 0.2,
              ease: "easeInOut",
            }}
          >
            <motion.p className="text-2xl text-muted-foreground mb-8 relative">
              <span className="text-6xl absolute -left-4 -top-2 text-violet-300">"</span>
              {testimonials[active].quote.split(" ").map((word, index) => (
                <motion.span
                  key={index}
                  initial={{
                    filter: "blur(10px)",
                    opacity: 0,
                    y: 5,
                  }}
                  animate={{
                    filter: "blur(0px)",
                    opacity: 1,
                    y: 0,
                  }}
                  transition={{
                    duration: 0.2,
                    ease: "easeInOut",
                    delay: 0.02 * index,
                  }}
                  className="inline-block"
                >
                  {word}&nbsp;
                </motion.span>
              ))}
              <span className="text-6xl text-violet-300 align-top">"</span>
            </motion.p>
            <div className="mt-6">
              <p className="text-xl font-semibold text-foreground">
                — {testimonials[active].name}
              </p>
              <p className="text-base text-muted-foreground mt-1">
                {testimonials[active].designation}
              </p>
              {getCompanyLogo(testimonials[active].designation) && (
                <div className="mt-4">
                  <Image
                    src={getCompanyLogo(testimonials[active].designation)}
                    alt={testimonials[active].designation.split('@')[1]?.trim() || ''}
                    width={120}
                    height={40}
                    className="object-contain h-10"
                  />
                </div>
              )}
            </div>
          </motion.div>
          <div className="flex gap-4 pt-12 md:pt-0">
            <button
              onClick={handlePrev}
              className="h-12 w-12 rounded-full bg-secondary flex items-center justify-center group/button"
            >
              <IconArrowLeft className="h-6 w-6 text-foreground group-hover/button:rotate-12 transition-transform duration-300" />
            </button>
            <button
              onClick={handleNext}
              className="h-12 w-12 rounded-full bg-secondary flex items-center justify-center group/button"
            >
              <IconArrowRight className="h-6 w-6 text-foreground group-hover/button:-rotate-12 transition-transform duration-300" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

