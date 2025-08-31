"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const loadingVariants = cva(
  "animate-spin rounded-full border-2 border-solid border-current border-r-transparent align-[-0.125em] motion-reduce:animate-[spin_1.5s_linear_infinite]",
  {
    variants: {
      size: {
        sm: "h-4 w-4 border-2",
        md: "h-6 w-6 border-2",
        lg: "h-8 w-8 border-2",
        xl: "h-12 w-12 border-4",
      },
      variant: {
        default: "text-primary",
        secondary: "text-secondary",
        muted: "text-muted-foreground",
        destructive: "text-destructive",
        success: "text-green-600",
        warning: "text-yellow-600",
      },
    },
    defaultVariants: {
      size: "md",
      variant: "default",
    },
  }
);

interface LoadingProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof loadingVariants> {
  text?: string;
  overlay?: boolean;
}

const Loading = React.forwardRef<HTMLDivElement, LoadingProps>(
  ({ className, size, variant, text, overlay, ...props }, ref) => {
    const spinner = (
      <div
        className={cn(loadingVariants({ size, variant }), className)}
        role="status"
        aria-label="Loading"
        {...props}
        ref={ref}
      />
    );

    if (overlay) {
      return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm">
          <div className="flex flex-col items-center space-y-2">
            {spinner}
            {text && (
              <p className="text-sm text-muted-foreground animate-pulse">{text}</p>
            )}
          </div>
        </div>
      );
    }

    if (text) {
      return (
        <div className="flex items-center space-x-2">
          {spinner}
          <span className="text-sm text-muted-foreground">{text}</span>
        </div>
      );
    }

    return spinner;
  }
);
Loading.displayName = "Loading";

// Skeleton loading component for content placeholders
const Skeleton = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn(
        "animate-pulse rounded-md bg-muted",
        className
      )}
      {...props}
    />
  );
});
Skeleton.displayName = "Skeleton";

// Pulse loading component for inline content
const Pulse = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & {
    children: React.ReactNode;
  }
>(({ className, children, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn("animate-pulse", className)}
      {...props}
    >
      {children}
    </div>
  );
});
Pulse.displayName = "Pulse";

// Dots loading animation
const LoadingDots = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof loadingVariants>
>(({ className, size, variant, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn("flex space-x-1", className)}
      {...props}
    >
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className={cn(
            "rounded-full animate-bounce",
            size === "sm" && "h-1 w-1",
            size === "md" && "h-2 w-2",
            size === "lg" && "h-3 w-3",
            size === "xl" && "h-4 w-4",
            variant === "default" && "bg-primary",
            variant === "secondary" && "bg-secondary",
            variant === "muted" && "bg-muted-foreground",
            variant === "destructive" && "bg-destructive",
            variant === "success" && "bg-green-600",
            variant === "warning" && "bg-yellow-600"
          )}
          style={{
            animationDelay: `${i * 0.1}s`,
          }}
        />
      ))}
    </div>
  );
});
LoadingDots.displayName = "LoadingDots";

export { Loading, Skeleton, Pulse, LoadingDots, loadingVariants };
export type { LoadingProps };