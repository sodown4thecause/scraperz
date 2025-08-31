"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { StatsCard, StatsGrid } from "@/components/ui/stats-card";
import { Loading, Skeleton } from "@/components/ui/loading";
import { useToast } from "@/hooks/use-toast";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from "recharts";
import {
  Activity,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  TrendingUp,
  Database,
  Globe,
  Cpu,
  HardDrive,
  Wifi,
  RefreshCw,
  Play,
  Pause,
  Settings,
  Download,
  Eye,
  BarChart3,
  PieChart as PieChartIcon,
  LineChart as LineChartIcon,
} from "lucide-react";

interface DashboardStats {
  totalJobs: number;
  activeJobs: number;
  completedJobs: number;
  failedJobs: number;
  successRate: number;
  avgExecutionTime: number;
  dataExtracted: number;
  systemHealth: {
    cpu: number;
    memory: number;
    disk: number;
    network: number;
  };
}

interface PerformanceData {
  timestamp: string;
  successRate: number;
  avgExecutionTime: number;
  dataQuality: number;
  jobsCompleted: number;
  cpuUsage: number;
  memoryUsage: number;
}

interface RecentJob {
  id: string;
  url: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  strategy: string;
  startTime: string;
  duration?: number;
  dataExtracted: number;
  qualityScore: number;
}

const EnhancedDashboard = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [performanceData, setPerformanceData] = useState<PerformanceData[]>([]);
  const [recentJobs, setRecentJobs] = useState<RecentJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const { toast } = useToast();

  // Mock data for demonstration
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        setStats({
          totalJobs: 1247,
          activeJobs: 8,
          completedJobs: 1198,
          failedJobs: 41,
          successRate: 96.2,
          avgExecutionTime: 2.4,
          dataExtracted: 45678,
          systemHealth: {
            cpu: 45,
            memory: 62,
            disk: 78,
            network: 89,
          },
        });

        setPerformanceData([
          { timestamp: '00:00', successRate: 94, avgExecutionTime: 2.8, dataQuality: 92, jobsCompleted: 45, cpuUsage: 40, memoryUsage: 55 },
          { timestamp: '04:00', successRate: 96, avgExecutionTime: 2.5, dataQuality: 94, jobsCompleted: 52, cpuUsage: 45, memoryUsage: 60 },
          { timestamp: '08:00', successRate: 98, avgExecutionTime: 2.2, dataQuality: 96, jobsCompleted: 68, cpuUsage: 50, memoryUsage: 65 },
          { timestamp: '12:00', successRate: 95, avgExecutionTime: 2.6, dataQuality: 93, jobsCompleted: 71, cpuUsage: 55, memoryUsage: 70 },
          { timestamp: '16:00', successRate: 97, avgExecutionTime: 2.3, dataQuality: 95, jobsCompleted: 63, cpuUsage: 48, memoryUsage: 62 },
          { timestamp: '20:00', successRate: 96, avgExecutionTime: 2.4, dataQuality: 94, jobsCompleted: 58, cpuUsage: 42, memoryUsage: 58 },
        ]);

        setRecentJobs([
          { id: '1', url: 'https://example.com', status: 'completed', strategy: 'Crawl4AI', startTime: '2024-01-15T10:30:00Z', duration: 2.3, dataExtracted: 156, qualityScore: 94 },
          { id: '2', url: 'https://news.site.com', status: 'running', strategy: 'Scrapy', startTime: '2024-01-15T10:28:00Z', dataExtracted: 89, qualityScore: 91 },
          { id: '3', url: 'https://ecommerce.com', status: 'completed', strategy: 'Adaptive', startTime: '2024-01-15T10:25:00Z', duration: 1.8, dataExtracted: 234, qualityScore: 97 },
          { id: '4', url: 'https://blog.example.org', status: 'failed', strategy: 'Crawl4AI', startTime: '2024-01-15T10:22:00Z', duration: 0.5, dataExtracted: 0, qualityScore: 0 },
          { id: '5', url: 'https://api.service.com', status: 'pending', strategy: 'Scrapy', startTime: '2024-01-15T10:20:00Z', dataExtracted: 0, qualityScore: 0 },
        ]);

        setLoading(false);
      } catch (error) {
        toast({
          title: "Error",
          description: "Failed to load dashboard data",
          variant: "destructive",
        });
        setLoading(false);
      }
    };

    fetchDashboardData();

    // Auto-refresh every 30 seconds
    const interval = autoRefresh ? setInterval(fetchDashboardData, 30000) : null;
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, toast]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'running': return <Clock className="h-4 w-4 text-blue-600" />;
      case 'failed': return <XCircle className="h-4 w-4 text-red-600" />;
      case 'pending': return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const variants = {
      completed: 'default' as const,
      running: 'secondary' as const,
      failed: 'destructive' as const,
      pending: 'outline' as const,
    };
    return variants[status as keyof typeof variants] || 'outline';
  };

  const chartColors = {
    primary: '#3b82f6',
    secondary: '#10b981',
    accent: '#f59e0b',
    muted: '#6b7280',
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-10 w-32" />
        </div>
        <StatsGrid>
          {[...Array(4)].map((_, i) => (
            <StatsCard
              key={i}
              title=""
              value=""
              loading={true}
            />
          ))}
        </StatsGrid>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Skeleton className="h-80" />
          <Skeleton className="h-80" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">
            Monitor your scraping operations and system performance
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant={autoRefresh ? "default" : "outline"}
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
            Auto Refresh
          </Button>
          <Button variant="outline" size="sm">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
        </div>
      </div>

      {/* Stats Overview */}
      <StatsGrid>
        <StatsCard
          title="Total Jobs"
          value={stats?.totalJobs.toLocaleString() || '0'}
          description="All time scraping jobs"
          icon={<Database className="h-4 w-4" />}
          trend={{ value: 12.5, label: "vs last month" }}
        />
        <StatsCard
          title="Active Jobs"
          value={stats?.activeJobs || 0}
          description="Currently running"
          icon={<Activity className="h-4 w-4" />}
          variant="info"
          badge={{ text: "Live", variant: "secondary" }}
        />
        <StatsCard
          title="Success Rate"
          value={`${stats?.successRate || 0}%`}
          description="Last 24 hours"
          icon={<CheckCircle className="h-4 w-4" />}
          variant="success"
          trend={{ value: 2.1, label: "vs yesterday" }}
        />
        <StatsCard
          title="Avg Execution Time"
          value={`${stats?.avgExecutionTime || 0}s`}
          description="Per job completion"
          icon={<Clock className="h-4 w-4" />}
          trend={{ value: -8.3, label: "improvement" }}
        />
      </StatsGrid>

      {/* Charts and Analytics */}
      <Tabs defaultValue="performance" className="space-y-4">
        <TabsList>
          <TabsTrigger value="performance">
            <LineChartIcon className="h-4 w-4 mr-2" />
            Performance
          </TabsTrigger>
          <TabsTrigger value="system">
            <BarChart3 className="h-4 w-4 mr-2" />
            System Health
          </TabsTrigger>
          <TabsTrigger value="jobs">
            <PieChartIcon className="h-4 w-4 mr-2" />
            Job Distribution
          </TabsTrigger>
        </TabsList>

        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Success Rate Trend</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Area
                      type="monotone"
                      dataKey="successRate"
                      stroke={chartColors.primary}
                      fill={chartColors.primary}
                      fillOpacity={0.1}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Execution Time & Quality</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="avgExecutionTime"
                      stroke={chartColors.secondary}
                      strokeWidth={2}
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="dataQuality"
                      stroke={chartColors.accent}
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="system" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>System Resources</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Cpu className="h-4 w-4" />
                      <span className="text-sm font-medium">CPU Usage</span>
                    </div>
                    <span className="text-sm text-muted-foreground">{stats?.systemHealth.cpu}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <HardDrive className="h-4 w-4" />
                      <span className="text-sm font-medium">Memory</span>
                    </div>
                    <span className="text-sm text-muted-foreground">{stats?.systemHealth.memory}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Database className="h-4 w-4" />
                      <span className="text-sm font-medium">Disk</span>
                    </div>
                    <span className="text-sm text-muted-foreground">{stats?.systemHealth.disk}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Wifi className="h-4 w-4" />
                      <span className="text-sm font-medium">Network</span>
                    </div>
                    <span className="text-sm text-muted-foreground">{stats?.systemHealth.network}%</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Resource Usage Over Time</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="cpuUsage" fill={chartColors.primary} />
                    <Bar dataKey="memoryUsage" fill={chartColors.secondary} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="jobs" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Jobs</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentJobs.map((job) => (
                  <div key={job.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      {getStatusIcon(job.status)}
                      <div>
                        <p className="font-medium">{job.url}</p>
                        <p className="text-sm text-muted-foreground">
                          {job.strategy} • {new Date(job.startTime).toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <Badge variant={getStatusBadge(job.status)}>
                        {job.status}
                      </Badge>
                      <div className="text-right">
                        <p className="text-sm font-medium">{job.dataExtracted} items</p>
                        <p className="text-xs text-muted-foreground">
                          Quality: {job.qualityScore}%
                        </p>
                      </div>
                      <Button variant="ghost" size="sm">
                        <Eye className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default EnhancedDashboard;