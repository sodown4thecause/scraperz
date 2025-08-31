"use client";

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Clock,
  CheckCircle,
  AlertTriangle,
  Zap,
  Globe,
} from 'lucide-react';

interface DataVisualizationProps {
  data: {
    performanceMetrics?: {
      totalJobs: number;
      successRate: number;
      avgProcessingTime: number;
      dataQuality: number;
      trendsData: Array<{
        date: string;
        jobs: number;
        success: number;
        avgTime: number;
      }>;
    };
    contentAnalytics?: {
      contentTypes: Array<{
        name: string;
        value: number;
        color: string;
      }>;
      qualityDistribution: Array<{
        range: string;
        count: number;
      }>;
      extractionStrategies: Array<{
        strategy: string;
        usage: number;
        successRate: number;
      }>;
    };
    systemHealth?: {
      cpuUsage: number;
      memoryUsage: number;
      diskUsage: number;
      networkLatency: number;
      activeConnections: number;
    };
    recentActivity?: Array<{
      timestamp: string;
      action: string;
      status: 'success' | 'error' | 'warning';
      duration: number;
    }>;
  };
}

const COLORS = {
  primary: '#3b82f6',
  success: '#10b981',
  warning: '#f59e0b',
  error: '#ef4444',
  secondary: '#6b7280',
  accent: '#8b5cf6',
};

const PIE_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'];

export default function DataVisualization({ data }: DataVisualizationProps) {
  const { performanceMetrics, contentAnalytics, systemHealth, recentActivity } = data;

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'error':
        return <AlertTriangle className="h-4 w-4 text-red-600" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      default:
        return <Activity className="h-4 w-4 text-blue-600" />;
    }
  };

  const getHealthColor = (percentage: number) => {
    if (percentage < 60) return 'text-green-600';
    if (percentage < 80) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getHealthVariant = (percentage: number): "default" | "destructive" | "secondary" | "outline" => {
    if (percentage < 60) return 'default';
    if (percentage < 80) return 'secondary';
    return 'destructive';
  };

  return (
    <div className="space-y-6">
      {/* Performance Overview */}
      {performanceMetrics && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Performance Trends */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Performance Trends
              </CardTitle>
              <CardDescription>
                Job execution and success rates over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={performanceMetrics.trendsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Area
                    type="monotone"
                    dataKey="jobs"
                    stackId="1"
                    stroke={COLORS.primary}
                    fill={COLORS.primary}
                    fillOpacity={0.6}
                    name="Total Jobs"
                  />
                  <Area
                    type="monotone"
                    dataKey="success"
                    stackId="2"
                    stroke={COLORS.success}
                    fill={COLORS.success}
                    fillOpacity={0.6}
                    name="Successful Jobs"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Processing Time Trends */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="h-5 w-5" />
                Processing Time Analysis
              </CardTitle>
              <CardDescription>
                Average processing time trends
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={performanceMetrics.trendsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip formatter={(value) => [formatDuration(value as number), 'Avg Time']} />
                  <Line
                    type="monotone"
                    dataKey="avgTime"
                    stroke={COLORS.accent}
                    strokeWidth={2}
                    dot={{ fill: COLORS.accent, strokeWidth: 2, r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Content Analytics */}
      {contentAnalytics && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Content Types Distribution */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Globe className="h-5 w-5" />
                Content Types
              </CardTitle>
              <CardDescription>
                Distribution of scraped content types
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={contentAnalytics.contentTypes}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {contentAnalytics.contentTypes.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-4 space-y-2">
                {contentAnalytics.contentTypes.map((type, index) => (
                  <div key={type.name} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: PIE_COLORS[index % PIE_COLORS.length] }}
                      />
                      <span className="text-sm">{type.name}</span>
                    </div>
                    <Badge variant="secondary">{type.value}</Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Quality Distribution */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Quality Distribution
              </CardTitle>
              <CardDescription>
                Data quality score ranges
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={contentAnalytics.qualityDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill={COLORS.primary} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Extraction Strategies */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Extraction Strategies
              </CardTitle>
              <CardDescription>
                Strategy usage and success rates
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {contentAnalytics.extractionStrategies.map((strategy, index) => (
                  <div key={strategy.strategy} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">{strategy.strategy}</span>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">{strategy.usage} uses</Badge>
                        <Badge
                          variant={strategy.successRate > 0.8 ? 'default' : strategy.successRate > 0.6 ? 'secondary' : 'destructive'}
                        >
                          {(strategy.successRate * 100).toFixed(0)}%
                        </Badge>
                      </div>
                    </div>
                    <Progress value={strategy.successRate * 100} className="h-2" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* System Health */}
      {systemHealth && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              System Health
            </CardTitle>
            <CardDescription>
              Real-time system performance metrics
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="text-center">
                <div className={`text-2xl font-bold ${getHealthColor(systemHealth.cpuUsage)}`}>
                  {systemHealth.cpuUsage}%
                </div>
                <div className="text-sm text-muted-foreground">CPU Usage</div>
                <Progress value={systemHealth.cpuUsage} className="mt-2 h-2" />
              </div>
              <div className="text-center">
                <div className={`text-2xl font-bold ${getHealthColor(systemHealth.memoryUsage)}`}>
                  {systemHealth.memoryUsage}%
                </div>
                <div className="text-sm text-muted-foreground">Memory</div>
                <Progress value={systemHealth.memoryUsage} className="mt-2 h-2" />
              </div>
              <div className="text-center">
                <div className={`text-2xl font-bold ${getHealthColor(systemHealth.diskUsage)}`}>
                  {systemHealth.diskUsage}%
                </div>
                <div className="text-sm text-muted-foreground">Disk Usage</div>
                <Progress value={systemHealth.diskUsage} className="mt-2 h-2" />
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {systemHealth.networkLatency}ms
                </div>
                <div className="text-sm text-muted-foreground">Latency</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {systemHealth.activeConnections}
                </div>
                <div className="text-sm text-muted-foreground">Connections</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recent Activity */}
      {recentActivity && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="h-5 w-5" />
              Recent Activity
            </CardTitle>
            <CardDescription>
              Latest scraping operations and system events
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {recentActivity.slice(0, 10).map((activity, index) => (
                <div key={index} className="flex items-center justify-between p-3 rounded-lg border">
                  <div className="flex items-center gap-3">
                    {getStatusIcon(activity.status)}
                    <div>
                      <p className="text-sm font-medium">{activity.action}</p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(activity.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <Badge variant={getHealthVariant(activity.duration / 1000)}>
                      {formatDuration(activity.duration)}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}