"use client";

import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { ListFilter, X } from "lucide-react";
import { nanoid } from "nanoid";
import * as React from "react";
import { AnimateChangeInHeight } from "@/components/ui/filters";
import {
  Filter,
  FilterOperator,
  FilterOption,
  FilterType,
  filterViewOptions,
  filterViewToFilterOptions,
} from "@/components/ui/filters";

interface ModelFiltersProps {
  onFilterChange?: (filters: Filter[]) => void;
}

export function ModelFilters({ onFilterChange }: ModelFiltersProps) {
  const [open, setOpen] = React.useState(false);
  const [selectedView, setSelectedView] = React.useState<FilterType | null>(
    null
  );
  const [commandInput, setCommandInput] = React.useState("");
  const commandInputRef = React.useRef<HTMLInputElement>(null);
  const [filters, setFilters] = React.useState<Filter[]>([]);

  React.useEffect(() => {
    if (onFilterChange) {
      onFilterChange(filters);
    }
  }, [filters, onFilterChange]);

  const activeFilterCount = filters.filter((filter) => filter.value?.length > 0).length;

  return (
    <div className="flex gap-2 flex-wrap items-center">
      <div className="flex items-center gap-1">
        <Popover
          open={open}
          onOpenChange={(open) => {
            setOpen(open);
            if (!open) {
              setTimeout(() => {
                setSelectedView(null);
                setCommandInput("");
              }, 200);
            }
          }}
        >
          <PopoverTrigger asChild>
            <Button
              variant="ghost"
              role="combobox"
              aria-expanded={open}
              size="sm"
              className="h-8 px-2 hover:bg-gray-100 flex items-center gap-1.5"
            >
              <ListFilter className="h-4 w-4 text-gray-600" />
              {activeFilterCount > 0 && (
                <span className="text-xs text-gray-600">({activeFilterCount})</span>
              )}
            </Button>
          </PopoverTrigger>
        <PopoverContent className="w-[200px] p-0">
          <AnimateChangeInHeight>
            <Command>
              <CommandInput
                placeholder={selectedView ? selectedView : "Filter..."}
                className="h-9"
                value={commandInput}
                onInputCapture={(e) => {
                  setCommandInput(e.currentTarget.value);
                }}
                ref={commandInputRef}
              />
              <CommandList>
                <CommandEmpty>No results found.</CommandEmpty>
                {selectedView ? (
                  <CommandGroup>
                    {filterViewToFilterOptions[selectedView].map(
                      (filter: FilterOption) => (
                        <CommandItem
                          className="group text-muted-foreground flex gap-2 items-center"
                          key={filter.name}
                          value={filter.name}
                          onSelect={(currentValue) => {
                            setFilters((prev) => [
                              ...prev,
                              {
                                id: nanoid(),
                                type: selectedView,
                                operator: FilterOperator.IS,
                                value: [currentValue],
                              },
                            ]);
                            setTimeout(() => {
                              setSelectedView(null);
                              setCommandInput("");
                            }, 200);
                            setOpen(false);
                          }}
                        >
                          {filter.icon}
                          <span className="text-accent-foreground">
                            {filter.name}
                          </span>
                          {filter.label && (
                            <span className="text-muted-foreground text-xs ml-auto">
                              {filter.label}
                            </span>
                          )}
                        </CommandItem>
                      )
                    )}
                  </CommandGroup>
                ) : (
                  filterViewOptions.map(
                    (group: FilterOption[], index: number) => (
                      <React.Fragment key={index}>
                        <CommandGroup>
                          {group.map((filter: FilterOption) => (
                            <CommandItem
                              className="group text-muted-foreground flex gap-2 items-center cursor-pointer"
                              key={filter.name}
                              value={filter.name}
                              onSelect={(currentValue) => {
                                setSelectedView(currentValue as FilterType);
                                setCommandInput("");
                                commandInputRef.current?.focus();
                              }}
                            >
                              {filter.icon}
                              <span className="text-accent-foreground">
                                {filter.name}
                              </span>
                            </CommandItem>
                          ))}
                        </CommandGroup>
                        {index < filterViewOptions.length - 1 && (
                          <CommandSeparator />
                        )}
                      </React.Fragment>
                    )
                  )
                )}
              </CommandList>
            </Command>
          </AnimateChangeInHeight>
        </PopoverContent>
      </Popover>
      {activeFilterCount > 0 && (
        <Button
          variant="ghost"
          size="sm"
          className="h-8 w-8 p-0 hover:bg-gray-100"
          onClick={() => setFilters([])}
          title="Clear all filters"
        >
          <X className="h-4 w-4 text-gray-600" />
        </Button>
      )}
      </div>
    </div>
  );
}
