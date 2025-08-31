"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
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
} from "lucide-react";

interface JobStatus {
  id: string;
  url: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  strategy: string;
  startTime: string;
  estimatedCompletion?: string;
  dataExtracted: number;
  qualityScore: number;
  antiDetectionTriggered: boolean;
}

interface SystemHealth {
  cpu: number;
  memory: number;
  disk: number;
  network: number;
  activeJobs: number;
  queuedJobs: number;
  totalJobsToday: number;
  successRate: number;
  avgResponseTime: number;
}

interface QualityMetrics {
  completeness: number;
  accuracy: number;
  relevance: number;
  structure: number;
  freshness: number;
}

interface PerformanceData {
  timestamp: string;
  successRate: number;
  avgExecutionTime: number;
  dataQuality: number;
  jobsCompleted: number;
}

const MonitoringDashboard = () => {
  const [activeJobs, setActiveJobs] = useState<JobStatus[]>([]);
  const [systemHealth, setSystemHealth] = useState<SystemHealth>({
    cpu: 0,
    memory: 0,
    disk: 0,
    network: 0,
    activeJobs: 0,
    queuedJobs: 0,
    totalJobsToday: 0,
    successRate: 0,
    avgResponseTime: 0,
  });
  const [performanceData, setPerformanceData] = useState<PerformanceData[]>([]);
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetrics>({
    completeness: 0,
    accuracy: 0,
    relevance: 0,
    structure: 0,
    freshness: 0,
  });
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // WebSocket connection for real-time updates
  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket(`ws://localhost:8000/monitoring/ws`);
      
      ws.onopen = () => {
        setIsConnected(true);
        console.log('Connected to monitoring WebSocket');
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setLastUpdate(new Date());
        
        switch (data.type) {
          case 'job_update':
            setActiveJobs(prev => {
              const updated = prev.filter(job => job.id !== data.job.id);
              if (data.job.status !== 'completed' && data.job.status !== 'failed') {
                updated.push(data.job);
              }
              return updated;
            });
            break;
          case 'system_health':
            setSystemHealth(data.health);
            break;
          case 'performance_update':
            setPerformanceData(prev => {
              const updated = [...prev, data.performance].slice(-20); // Keep last 20 points
              return updated;
            });
            break;
          case 'quality_metrics':
            setQualityMetrics(data.metrics);
            break;
        }
      };
      
      ws.onclose = () => {
        setIsConnected(false);
        console.log('Disconnected from monitoring WebSocket');
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };
      
      return ws;
    };

    const ws = connectWebSocket();
    
    return () => {
      ws.close();
    };
  }, []);

  // Fetch functions for different data types
  const fetchPerformanceData = async () => {
    try {
      const response = await fetch('http://localhost:8000/monitoring/performance');
      if (response.ok) {
        const data = await response.json();
        setPerformanceData(data.performance || []);
      }
    } catch (error) {
      console.error('Failed to fetch performance data:', error);
    }
  };

  const fetchQualityMetrics = async () => {
    try {
      const response = await fetch('http://localhost:8000/monitoring/quality');
      if (response.ok) {
        const data = await response.json();
        setQualityMetrics(data.quality || {});
      }
    } catch (error) {
      console.error('Failed to fetch quality metrics:', error);
    }
  };

  const fetchAnalytics = async () => {
    try {
      const response = await fetch('http://localhost:8000/monitoring/analytics');
      if (response.ok) {
        const data = await response.json();
        // Handle analytics data if needed
      }
    } catch (error) {
      console.error('Failed to fetch analytics:', error);
    }
  };

  // Fallback polling for when WebSocket is not available
  useEffect(() => {
    if (!isConnected) {
      const interval = setInterval(async () => {
        try {
          const response = await fetch('http://localhost:8000/monitoring/status');
          if (response.ok) {
            const data = await response.json();
            setActiveJobs(data.activeJobs);
            setSystemHealth(data.systemHealth);
            setPerformanceData(prev => [...prev, data.performance].slice(-20));
            setQualityMetrics(data.qualityMetrics);
            setLastUpdate(new Date());
          }
        } catch (error) {
          console.error('Failed to fetch monitoring data:', error);
        }
      }, 5000);
      
      return () => clearInterval(interval);
    }
  }, [isConnected]);

  // Initial data fetch
   useEffect(() => {
     fetchPerformanceData();
     fetchQualityMetrics();
     fetchAnalytics();
   }, []);
 
   const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-500';
      case 'running': return 'bg-blue-500';
      case 'failed': return 'bg-red-500';
      case 'pending': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="h-4 w-4" />;
      case 'running': return <Activity className="h-4 w-4" />;
      case 'failed': return <XCircle className="h-4 w-4" />;
      case 'pending': return <Clock className="h-4 w-4" />;
      default: return <AlertTriangle className="h-4 w-4" />;
    }
  };

  const getHealthColor = (value: number) => {
    if (value < 50) return 'text-green-600';
    if (value < 80) return 'text-yellow-600';
    return 'text-red-600';
  };

  const strategyColors = {
    crawl4ai: '#8884d8',
    scrapy: '#82ca9d',
    scrapegraph: '#ffc658',
    hybrid: '#ff7300',
  };

  return (
    <div className="space-y-6">
      {/* Header with connection status */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Monitoring Dashboard</h1>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${
              isConnected ? 'bg-green-500' : 'bg-red-500'
            }`} />
            <span className="text-sm text-gray-600">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <span className="text-sm text-gray-500">
            Last update: {lastUpdate.toLocaleTimeString()}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => window.location.reload()}
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* System Health Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemHealth.cpu}%</div>
            <Progress value={systemHealth.cpu} className="mt-2" />
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemHealth.memory}%</div>
            <Progress value={systemHealth.memory} className="mt-2" />
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Jobs</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemHealth.activeJobs}</div>
            <p className="text-xs text-muted-foreground">
              {systemHealth.queuedJobs} queued
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemHealth.successRate}%</div>
            <p className="text-xs text-muted-foreground">
              {systemHealth.totalJobsToday} jobs today
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Dashboard Tabs */}
      <Tabs defaultValue="jobs" className="space-y-4">
        <TabsList>
          <TabsTrigger value="jobs">Active Jobs</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="quality">Data Quality</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        {/* Active Jobs Tab */}
        <TabsContent value="jobs" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Active Scraping Jobs</CardTitle>
            </CardHeader>
            <CardContent>
              {activeJobs.length === 0 ? (
                <p className="text-center text-gray-500 py-8">No active jobs</p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Status</TableHead>
                      <TableHead>URL</TableHead>
                      <TableHead>Strategy</TableHead>
                      <TableHead>Progress</TableHead>
                      <TableHead>Quality</TableHead>
                      <TableHead>Data Extracted</TableHead>
                      <TableHead>Started</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {activeJobs.map((job) => (
                      <TableRow key={job.id}>
                        <TableCell>
                          <div className="flex items-center space-x-2">
                            {getStatusIcon(job.status)}
                            <Badge variant="outline" className={getStatusColor(job.status)}>
                              {job.status}
                            </Badge>
                            {job.antiDetectionTriggered && (
                              <AlertTriangle className="h-4 w-4 text-yellow-500" title="Anti-detection triggered" />
                            )}
                          </div>
                        </TableCell>
                        <TableCell className="max-w-xs truncate">{job.url}</TableCell>
                        <TableCell>
                          <Badge variant="secondary">{job.strategy}</Badge>
                        </TableCell>
                        <TableCell>
                          <div className="w-20">
                            <Progress value={job.progress} />
                            <span className="text-xs text-gray-500">{job.progress}%</span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className="text-sm">{job.qualityScore}/100</div>
                        </TableCell>
                        <TableCell>{job.dataExtracted} items</TableCell>
                        <TableCell>{new Date(job.startTime).toLocaleTimeString()}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Success Rate Trend</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="successRate" stroke="#8884d8" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Execution Time</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="avgExecutionTime" stroke="#82ca9d" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Data Quality Tab */}
        <TabsContent value="quality" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Quality Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Completeness</span>
                    <span>{qualityMetrics.completeness}%</span>
                  </div>
                  <Progress value={qualityMetrics.completeness} />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Accuracy</span>
                    <span>{qualityMetrics.accuracy}%</span>
                  </div>
                  <Progress value={qualityMetrics.accuracy} />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Relevance</span>
                    <span>{qualityMetrics.relevance}%</span>
                  </div>
                  <Progress value={qualityMetrics.relevance} />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Structure</span>
                    <span>{qualityMetrics.structure}%</span>
                  </div>
                  <Progress value={qualityMetrics.structure} />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Freshness</span>
                    <span>{qualityMetrics.freshness}%</span>
                  </div>
                  <Progress value={qualityMetrics.freshness} />
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Data Quality Trend</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="dataQuality" stroke="#ffc658" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Jobs Completed</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="jobsCompleted" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Strategy Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Crawl4AI', value: 35, fill: strategyColors.crawl4ai },
                        { name: 'Scrapy', value: 25, fill: strategyColors.scrapy },
                        { name: 'ScrapeGraph', value: 30, fill: strategyColors.scrapegraph },
                        { name: 'Hybrid', value: 10, fill: strategyColors.hybrid },
                      ]}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                      label
                    />
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MonitoringDashboard;