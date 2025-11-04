"use client";

import * as React from "react";
import { X } from "lucide-react";
import { motion } from "framer-motion";

export enum FilterType {
  STATUS = "Status",
  MODEL_NAME = "Model Name",
  COMPLETION = "Completion",
}

export enum FilterOperator {
  IS = "is",
  IS_NOT = "is not",
  CONTAINS = "contains",
}

export interface Filter {
  id: string;
  type: FilterType;
  operator: FilterOperator;
  value: string[];
}

export interface FilterOption {
  name: string;
  icon?: React.ReactNode;
  label?: string;
}

export const filterViewToFilterOptions: Record<FilterType, FilterOption[]> = {
  [FilterType.STATUS]: [
    { name: "Complete" },
    { name: "Incomplete" },
  ],
  [FilterType.MODEL_NAME]: [],
  [FilterType.COMPLETION]: [
    { name: "Ready to Use" },
    { name: "Needs Host" },
  ],
};

export const filterViewOptions: FilterOption[][] = [
  [
    { name: FilterType.STATUS },
    { name: FilterType.COMPLETION },
  ],
];

export function AnimateChangeInHeight({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial={false}
      animate={{ height: "auto" }}
      transition={{ duration: 0.2 }}
    >
      {children}
    </motion.div>
  );
}

interface FiltersProps {
  filters: Filter[];
  setFilters: React.Dispatch<React.SetStateAction<Filter[]>>;
}

export default function Filters({ filters, setFilters }: FiltersProps) {
  const removeFilter = (id: string) => {
    setFilters((prev) => prev.filter((f) => f.id !== id));
  };

  return (
    <>
      {filters.map((filter) => (
        <div
          key={filter.id}
          className="flex items-center gap-1 px-2 py-1 bg-violet-100 text-violet-700 rounded-md text-xs"
        >
          <span className="font-medium">{filter.type}:</span>
          <span>{filter.value.join(", ")}</span>
          <button
            onClick={() => removeFilter(filter.id)}
            className="ml-1 hover:bg-violet-200 rounded-full p-0.5"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      ))}
    </>
  );
}

