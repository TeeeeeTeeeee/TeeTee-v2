"use client";

import React from "react";
import Image from "next/image";

export const BrandCarousel = () => {
  // All unique brand logos
  const allBrands = [
    { name: 'Stripe', logo: '/logos/stripe.png' },
    { name: 'Notion', logo: '/logos/notion.png' },
    { name: 'Figma', logo: '/logos/figma.png' },
    { name: 'Databricks', logo: '/logos/databricks.png' },
    { name: 'GitHub', logo: '/logos/github.png' },
    { name: 'OpenAI', logo: '/logos/openAI.png' },
    { name: 'MongoDB', logo: '/logos/mongodb.png' },
    { name: 'Vercel', logo: '/logos/vercel.png' },
    { name: 'Zapier', logo: '/logos/zapier.png' },
    { name: 'GitLab', logo: '/logos/gitlab.png' },
    { name: 'Datadog', logo: '/logos/datadog.png' },
    { name: 'Atlassian', logo: '/logos/atlassian.png' },
    { name: 'Elasic', logo: '/logos/elastic.png' },
    { name: 'Kubernetes', logo: '/logos/kubernetes.png' },
    { name: 'Pagerduty', logo: '/logos/pagerduty.png' },
    { name: 'Terraform', logo: '/logos/terraform.png' },
    { name: 'Newrelic', logo: '/logos/newrelic.png' },
  ];

  // Distribute brands across 3 rows - 5 brands each
  const brandsRow1 = [allBrands[0], allBrands[1], allBrands[2], allBrands[3], allBrands[4]];
  const brandsRow2 = [allBrands[5], allBrands[6], allBrands[16], allBrands[8], allBrands[9]];
  const brandsRow3 = [allBrands[10], allBrands[11], allBrands[12], allBrands[13], allBrands[14]];

  return (
    <section className="py-16 bg-gradient-to-r from-violet-100/40 via-purple-100/30 to-violet-100/40">
      {/* Title */}
      <div className="text-center mb-8">
        <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wider">
          Brands that trust us
        </h3>
      </div>
      
      <div className="overflow-hidden">
        <style jsx>{`
          @keyframes scroll {
            0% {
              transform: translateX(0);
            }
            100% {
              transform: translateX(-50%);
            }
          }
          .animate-scroll {
            animation: scroll 30s linear infinite;
          }
          .animate-scroll:hover {
            animation-play-state: paused;
          }
        `}</style>
        
        {/* Row 1 */}
        <div className="flex mb-8">
          <div className="flex animate-scroll">
            {[...brandsRow1, ...brandsRow1, ...brandsRow1, ...brandsRow1].map((brand, index) => (
              <div
                key={index}
                className="flex items-center justify-center min-w-[300px] h-24 mx-8"
              >
                <div className="relative w-32 h-12">
                  <Image
                    src={brand.logo}
                    alt={brand.name}
                    fill
                    className="object-contain"
                    sizes="128px"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Row 2 */}
        <div className="flex mb-8">
          <div className="flex animate-scroll" style={{ animationDuration: '35s', animationDirection: 'reverse' }}>
            {[...brandsRow2, ...brandsRow2, ...brandsRow2, ...brandsRow2].map((brand, index) => (
              <div
                key={index}
                className="flex items-center justify-center min-w-[300px] h-24 mx-8"
              >
                <div className="relative w-32 h-12">
                  <Image
                    src={brand.logo}
                    alt={brand.name}
                    fill
                    className="object-contain"
                    sizes="128px"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Row 3 */}
        <div className="flex">
          <div className="flex animate-scroll" style={{ animationDuration: '32s' }}>
            {[...brandsRow3, ...brandsRow3, ...brandsRow3, ...brandsRow3].map((brand, index) => (
              <div
                key={index}
                className="flex items-center justify-center min-w-[300px] h-24 mx-8"
              >
                <div className="relative w-32 h-12">
                  <Image
                    src={brand.logo}
                    alt={brand.name}
                    fill
                    className="object-contain"
                    sizes="128px"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

