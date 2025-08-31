"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Calendar } from "@/components/ui/calendar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Search,
  Filter,
  Download,
  Eye,
  Trash2,
  Calendar as CalendarIcon,
  MoreHorizontal,
  RefreshCw,
  FileText,
  Database,
  Image,
  BarChart3,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
} from "lucide-react";
import { format } from "date-fns";
import { cn } from "@/lib/utils";

interface ScrapingJob {
  id: string;
  url: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  strategy: string;
  startTime: string;
  endTime?: string;
  duration?: number;
  dataExtracted: number;
  qualityScore: number;
  errorMessage?: string;
  extractedData?: any;
  metadata: {
    userAgent: string;
    contentType: string;
    responseSize: number;
    antiDetectionTriggered: boolean;
  };
}

interface FilterOptions {
  status: string;
  strategy: string;
  dateRange: {
    from?: Date;
    to?: Date;
  };
  qualityThreshold: number;
}

const JobHistoryManager = () => {
  const [jobs, setJobs] = useState<ScrapingJob[]>([]);
  const [filteredJobs, setFilteredJobs] = useState<ScrapingJob[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [filters, setFilters] = useState<FilterOptions>({
    status: "all",
    strategy: "all",
    dateRange: {},
    qualityThreshold: 0,
  });
  const [sortBy, setSortBy] = useState("startTime");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [selectedJob, setSelectedJob] = useState<ScrapingJob | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);

  // Fetch jobs from API
  const fetchJobs = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/jobs');
      if (response.ok) {
        const data = await response.json();
        setJobs(data.jobs || []);
      }
    } catch (error) {
      console.error('Failed to fetch jobs:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
  }, []);

  // Apply filters and search
  useEffect(() => {
    let filtered = jobs.filter(job => {
      // Search filter
      const matchesSearch = job.url.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           job.id.toLowerCase().includes(searchTerm.toLowerCase());
      
      // Status filter
      const matchesStatus = filters.status === "all" || job.status === filters.status;
      
      // Strategy filter
      const matchesStrategy = filters.strategy === "all" || job.strategy === filters.strategy;
      
      // Date range filter
      const jobDate = new Date(job.startTime);
      const matchesDateRange = (!filters.dateRange.from || jobDate >= filters.dateRange.from) &&
                              (!filters.dateRange.to || jobDate <= filters.dateRange.to);
      
      // Quality threshold filter
      const matchesQuality = job.qualityScore >= filters.qualityThreshold;
      
      return matchesSearch && matchesStatus && matchesStrategy && matchesDateRange && matchesQuality;
    });

    // Apply sorting
    filtered.sort((a, b) => {
      let aValue = a[sortBy as keyof ScrapingJob];
      let bValue = b[sortBy as keyof ScrapingJob];
      
      if (typeof aValue === 'string') aValue = aValue.toLowerCase();
      if (typeof bValue === 'string') bValue = bValue.toLowerCase();
      
      if (aValue < bValue) return sortOrder === "asc" ? -1 : 1;
      if (aValue > bValue) return sortOrder === "asc" ? 1 : -1;
      return 0;
    });

    setFilteredJobs(filtered);
    setCurrentPage(1);
  }, [jobs, searchTerm, filters, sortBy, sortOrder]);

  // Export functions
  const exportToCSV = () => {
    const headers = ['ID', 'URL', 'Status', 'Strategy', 'Start Time', 'Duration', 'Data Extracted', 'Quality Score'];
    const csvContent = [
      headers.join(','),
      ...filteredJobs.map(job => [
        job.id,
        `"${job.url}"`,
        job.status,
        job.strategy,
        job.startTime,
        job.duration || 0,
        job.dataExtracted,
        job.qualityScore
      ].join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `scraping-jobs-${format(new Date(), 'yyyy-MM-dd')}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportToJSON = () => {
    const jsonContent = JSON.stringify(filteredJobs, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `scraping-jobs-${format(new Date(), 'yyyy-MM-dd')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Delete job
  const deleteJob = async (jobId: string) => {
    try {
      const response = await fetch(`/api/jobs/${jobId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        setJobs(prev => prev.filter(job => job.id !== jobId));
      }
    } catch (error) {
      console.error('Failed to delete job:', error);
    }
  };

  // Status badge component
  const StatusBadge = ({ status }: { status: string }) => {
    const variants = {
      pending: { variant: "secondary" as const, icon: Clock },
      running: { variant: "default" as const, icon: RefreshCw },
      completed: { variant: "default" as const, icon: CheckCircle },
      failed: { variant: "destructive" as const, icon: XCircle },
    };
    
    const config = variants[status as keyof typeof variants] || variants.pending;
    const Icon = config.icon;
    
    return (
      <Badge variant={config.variant} className="flex items-center gap-1">
        <Icon className="h-3 w-3" />
        {status}
      </Badge>
    );
  };

  // Pagination
  const totalPages = Math.ceil(filteredJobs.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedJobs = filteredJobs.slice(startIndex, startIndex + itemsPerPage);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Job History</h2>
          <p className="text-muted-foreground">
            Manage and analyze your scraping job history
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={fetchJobs} variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuItem onClick={exportToCSV}>
                <FileText className="h-4 w-4 mr-2" />
                Export as CSV
              </DropdownMenuItem>
              <DropdownMenuItem onClick={exportToJSON}>
                <Database className="h-4 w-4 mr-2" />
                Export as JSON
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Filters and Search */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Filters & Search</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search jobs..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-9"
              />
            </div>

            {/* Status Filter */}
            <Select
              value={filters.status}
              onValueChange={(value) => setFilters(prev => ({ ...prev, status: value }))}
            >
              <SelectTrigger>
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="running">Running</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
              </SelectContent>
            </Select>

            {/* Strategy Filter */}
            <Select
              value={filters.strategy}
              onValueChange={(value) => setFilters(prev => ({ ...prev, strategy: value }))}
            >
              <SelectTrigger>
                <SelectValue placeholder="Filter by strategy" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Strategies</SelectItem>
                <SelectItem value="crawl4ai">Crawl4AI</SelectItem>
                <SelectItem value="scrapy">Scrapy</SelectItem>
                <SelectItem value="ai_powered">AI Powered</SelectItem>
              </SelectContent>
            </Select>

            {/* Date Range Filter */}
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" className="justify-start text-left font-normal">
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {filters.dateRange.from ? (
                    filters.dateRange.to ? (
                      <>
                        {format(filters.dateRange.from, "LLL dd, y")} -{" "}
                        {format(filters.dateRange.to, "LLL dd, y")}
                      </>
                    ) : (
                      format(filters.dateRange.from, "LLL dd, y")
                    )
                  ) : (
                    <span>Pick a date range</span>
                  )}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar
                  initialFocus
                  mode="range"
                  defaultMonth={filters.dateRange.from}
                  selected={filters.dateRange}
                  onSelect={(range) => setFilters(prev => ({ ...prev, dateRange: range || {} }))}
                  numberOfMonths={2}
                />
              </PopoverContent>
            </Popover>
          </div>
        </CardContent>
      </Card>

      {/* Jobs Table */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <CardTitle>Jobs ({filteredJobs.length})</CardTitle>
            <div className="flex gap-2">
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-40">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="startTime">Start Time</SelectItem>
                  <SelectItem value="status">Status</SelectItem>
                  <SelectItem value="qualityScore">Quality Score</SelectItem>
                  <SelectItem value="dataExtracted">Data Extracted</SelectItem>
                </SelectContent>
              </Select>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSortOrder(prev => prev === "asc" ? "desc" : "asc")}
              >
                {sortOrder === "asc" ? "↑" : "↓"}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex justify-center py-8">
              <RefreshCw className="h-6 w-6 animate-spin" />
            </div>
          ) : (
            <>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>URL</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Strategy</TableHead>
                    <TableHead>Start Time</TableHead>
                    <TableHead>Duration</TableHead>
                    <TableHead>Data</TableHead>
                    <TableHead>Quality</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {paginatedJobs.map((job) => (
                    <TableRow key={job.id}>
                      <TableCell className="max-w-xs truncate" title={job.url}>
                        {job.url}
                      </TableCell>
                      <TableCell>
                        <StatusBadge status={job.status} />
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">{job.strategy}</Badge>
                      </TableCell>
                      <TableCell>
                        {format(new Date(job.startTime), "MMM dd, HH:mm")}
                      </TableCell>
                      <TableCell>
                        {job.duration ? `${Math.round(job.duration / 1000)}s` : "-"}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1">
                          <Database className="h-4 w-4" />
                          {job.dataExtracted}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1">
                          <BarChart3 className="h-4 w-4" />
                          {job.qualityScore.toFixed(1)}
                        </div>
                      </TableCell>
                      <TableCell>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="sm">
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent>
                            <DropdownMenuItem onClick={() => setSelectedJob(job)}>
                              <Eye className="h-4 w-4 mr-2" />
                              View Details
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem
                              onClick={() => deleteJob(job.id)}
                              className="text-destructive"
                            >
                              <Trash2 className="h-4 w-4 mr-2" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex justify-center gap-2 mt-4">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                    disabled={currentPage === 1}
                  >
                    Previous
                  </Button>
                  <span className="flex items-center px-3 text-sm">
                    Page {currentPage} of {totalPages}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                    disabled={currentPage === totalPages}
                  >
                    Next
                  </Button>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* Job Details Dialog */}
      <Dialog open={!!selectedJob} onOpenChange={() => setSelectedJob(null)}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Job Details</DialogTitle>
            <DialogDescription>
              Detailed information about the scraping job
            </DialogDescription>
          </DialogHeader>
          {selectedJob && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold mb-2">Basic Information</h4>
                  <div className="space-y-2 text-sm">
                    <div><strong>ID:</strong> {selectedJob.id}</div>
                    <div><strong>URL:</strong> {selectedJob.url}</div>
                    <div><strong>Status:</strong> <StatusBadge status={selectedJob.status} /></div>
                    <div><strong>Strategy:</strong> {selectedJob.strategy}</div>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Performance Metrics</h4>
                  <div className="space-y-2 text-sm">
                    <div><strong>Start Time:</strong> {format(new Date(selectedJob.startTime), "PPpp")}</div>
                    {selectedJob.endTime && (
                      <div><strong>End Time:</strong> {format(new Date(selectedJob.endTime), "PPpp")}</div>
                    )}
                    <div><strong>Duration:</strong> {selectedJob.duration ? `${Math.round(selectedJob.duration / 1000)}s` : "N/A"}</div>
                    <div><strong>Data Extracted:</strong> {selectedJob.dataExtracted} items</div>
                    <div><strong>Quality Score:</strong> {selectedJob.qualityScore.toFixed(2)}</div>
                  </div>
                </div>
              </div>
              
              {selectedJob.metadata && (
                <div>
                  <h4 className="font-semibold mb-2">Metadata</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div><strong>User Agent:</strong> {selectedJob.metadata.userAgent}</div>
                    <div><strong>Content Type:</strong> {selectedJob.metadata.contentType}</div>
                    <div><strong>Response Size:</strong> {selectedJob.metadata.responseSize} bytes</div>
                    <div><strong>Anti-Detection:</strong> {selectedJob.metadata.antiDetectionTriggered ? "Triggered" : "Not Triggered"}</div>
                  </div>
                </div>
              )}
              
              {selectedJob.errorMessage && (
                <div>
                  <h4 className="font-semibold mb-2 text-destructive">Error Message</h4>
                  <div className="bg-destructive/10 p-3 rounded text-sm">
                    {selectedJob.errorMessage}
                  </div>
                </div>
              )}
              
              {selectedJob.extractedData && (
                <div>
                  <h4 className="font-semibold mb-2">Extracted Data Preview</h4>
                  <pre className="bg-muted p-3 rounded text-xs overflow-x-auto max-h-40">
                    {JSON.stringify(selectedJob.extractedData, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default JobHistoryManager;