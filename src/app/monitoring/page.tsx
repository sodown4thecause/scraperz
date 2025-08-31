"use client";

import Header from "@/components/Header";
import MonitoringDashboard from "@/components/MonitoringDashboard";

export default function MonitoringPage() {
  return (
    <div>
      <Header />
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900">Real-time Monitoring Dashboard</h1>
          <p className="mt-2 text-gray-600">
            Monitor scraping jobs, system health, and data quality metrics in real-time.
          </p>
        </div>
        <MonitoringDashboard />
      </main>
    </div>
  );
}