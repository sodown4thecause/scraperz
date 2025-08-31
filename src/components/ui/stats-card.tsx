"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

const statsCardVariants = cva(
  "transition-all duration-200 hover:shadow-md",
  {
    variants: {
      variant: {
        default: "border-border",
        success: "border-green-200 bg-green-50/50 dark:border-green-800 dark:bg-green-950/50",
        warning: "border-yellow-200 bg-yellow-50/50 dark:border-yellow-800 dark:bg-yellow-950/50",
        destructive: "border-red-200 bg-red-50/50 dark:border-red-800 dark:bg-red-950/50",
        info: "border-blue-200 bg-blue-50/50 dark:border-blue-800 dark:bg-blue-950/50",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

interface StatsCardProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof statsCardVariants> {
  title: string;
  value: string | number;
  description?: string;
  icon?: React.ReactNode;
  trend?: {
    value: number;
    label?: string;
    period?: string;
  };
  badge?: {
    text: string;
    variant?: "default" | "secondary" | "destructive" | "outline";
  };
  loading?: boolean;
}

const StatsCard = React.forwardRef<HTMLDivElement, StatsCardProps>(
  ({
    className,
    variant,
    title,
    value,
    description,
    icon,
    trend,
    badge,
    loading = false,
    ...props
  }, ref) => {
    const getTrendIcon = (trendValue: number) => {
      if (trendValue > 0) return <TrendingUp className="h-3 w-3" />;
      if (trendValue < 0) return <TrendingDown className="h-3 w-3" />;
      return <Minus className="h-3 w-3" />;
    };

    const getTrendColor = (trendValue: number) => {
      if (trendValue > 0) return "text-green-600 dark:text-green-400";
      if (trendValue < 0) return "text-red-600 dark:text-red-400";
      return "text-muted-foreground";
    };

    if (loading) {
      return (
        <Card className={cn(statsCardVariants({ variant }), className)} ref={ref} {...props}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <div className="h-4 w-24 bg-muted animate-pulse rounded" />
            {icon && (
              <div className="h-4 w-4 bg-muted animate-pulse rounded" />
            )}
          </CardHeader>
          <CardContent>
            <div className="h-8 w-16 bg-muted animate-pulse rounded mb-2" />
            <div className="h-3 w-32 bg-muted animate-pulse rounded" />
          </CardContent>
        </Card>
      );
    }

    return (
      <Card className={cn(statsCardVariants({ variant }), className)} ref={ref} {...props}>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
          <div className="flex items-center space-x-2">
            {badge && (
              <Badge variant={badge.variant || "default"} className="text-xs">
                {badge.text}
              </Badge>
            )}
            {icon && (
              <div className="text-muted-foreground">
                {icon}
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-baseline justify-between">
            <div className="text-2xl font-bold">{value}</div>
            {trend && (
              <div className={cn(
                "flex items-center space-x-1 text-xs font-medium",
                getTrendColor(trend.value)
              )}>
                {getTrendIcon(trend.value)}
                <span>
                  {Math.abs(trend.value)}%
                  {trend.label && ` ${trend.label}`}
                </span>
              </div>
            )}
          </div>
          {description && (
            <p className="text-xs text-muted-foreground mt-1">
              {description}
              {trend?.period && (
                <span className="ml-1">• {trend.period}</span>
              )}
            </p>
          )}
        </CardContent>
      </Card>
    );
  }
);
StatsCard.displayName = "StatsCard";

// Grid container for stats cards
const StatsGrid = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & {
    cols?: 1 | 2 | 3 | 4;
  }
>(({ className, cols = 4, children, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn(
        "grid gap-4",
        cols === 1 && "grid-cols-1",
        cols === 2 && "grid-cols-1 md:grid-cols-2",
        cols === 3 && "grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
        cols === 4 && "grid-cols-1 md:grid-cols-2 lg:grid-cols-4",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
});
StatsGrid.displayName = "StatsGrid";

export { StatsCard, StatsGrid, statsCardVariants };
export type { StatsCardProps };