"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { 
  Download, 
  Copy, 
  Eye, 
  FileText, 
  Image, 
  Table as TableIcon, 
  Code,
  Clock,
  CheckCircle,
  AlertCircle,
  TrendingUp
} from 'lucide-react';

interface ResultsDisplayProps {
  data: {
    id?: string;
    url?: string;
    extractedData?: any;
    metadata?: {
      timestamp?: string;
      processingTime?: number;
      strategy?: string;
      contentType?: string;
      qualityScore?: number;
      tokensUsed?: number;
    };
    images?: Array<{
      url: string;
      alt?: string;
      description?: string;
    }>;
    tables?: Array<{
      headers: string[];
      rows: string[][];
      caption?: string;
    }>;
    structuredData?: any;
    rawHtml?: string;
  } | any;
}

export default function ResultsDisplay({ data }: ResultsDisplayProps) {
  const [activeTab, setActiveTab] = useState('extracted');
  const { toast } = useToast();

  if (!data) {
    return null;
  }

  const handleCopy = async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      toast({
        title: "Copied to clipboard",
        description: "Content has been copied successfully.",
      });
    } catch (error) {
      toast({
        title: "Copy failed",
        description: "Failed to copy content to clipboard.",
        variant: "destructive",
      });
    }
  };

  const handleDownload = (content: string, filename: string, type: string = 'application/json') => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast({
      title: "Download started",
      description: `${filename} is being downloaded.`,
    });
  };

  const formatJson = (obj: any) => JSON.stringify(obj, null, 2);

  const getQualityColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getQualityLabel = (score: number) => {
    if (score >= 0.8) return 'High';
    if (score >= 0.6) return 'Medium';
    return 'Low';
  };

  // Handle legacy data format (simple array or object)
  const isLegacyFormat = !data.extractedData && !data.metadata;
  const displayData = isLegacyFormat ? data : data.extractedData || data;
  const isTableData = Array.isArray(displayData) && displayData.length > 0 && typeof displayData[0] === 'object' && displayData[0] !== null;

  // Legacy simple display for backward compatibility
  if (isLegacyFormat) {
    if (isTableData) {
      const headers = Object.keys(displayData[0]);
      return (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                  Scraping Results
                </CardTitle>
                <CardDescription>
                  {displayData.length} records extracted
                </CardDescription>
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleCopy(formatJson(displayData))}
                >
                  <Copy className="h-4 w-4 mr-2" />
                  Copy
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleDownload(
                    formatJson(displayData),
                    `scraped-data-${Date.now()}.json`
                  )}
                >
                  <Download className="h-4 w-4 mr-2" />
                  Download
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-96 w-full">
              <Table>
                <TableHeader>
                  <TableRow>
                    {headers.map((header) => (
                      <TableHead key={header}>{header}</TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {displayData.map((row: any, rowIndex: number) => (
                    <TableRow key={rowIndex}>
                      {headers.map((header) => (
                        <TableCell key={`${rowIndex}-${header}`}>
                          {typeof row[header] === 'object' ? JSON.stringify(row[header]) : String(row[header])}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </ScrollArea>
          </CardContent>
        </Card>
      );
    }

    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle className="h-5 w-5 text-green-600" />
                Scraping Results
              </CardTitle>
              <CardDescription>
                Raw extracted data
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleCopy(formatJson(displayData))}
              >
                <Copy className="h-4 w-4 mr-2" />
                Copy
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleDownload(
                  formatJson(displayData),
                  `scraped-data-${Date.now()}.json`
                )}
              >
                <Download className="h-4 w-4 mr-2" />
                Download
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-96 w-full">
            <pre className="text-sm bg-muted p-4 rounded-lg overflow-auto">
              {formatJson(displayData)}
            </pre>
          </ScrollArea>
        </CardContent>
      </Card>
    );
  }

  // Enhanced display for new format
  return (
    <div className="space-y-6">
      {/* Results Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle className="h-5 w-5 text-green-600" />
                Extraction Complete
              </CardTitle>
              <CardDescription className="mt-1">
                {data.url ? `Data successfully extracted from ${data.url}` : 'Data extraction completed'}
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleCopy(formatJson(displayData))}
              >
                <Copy className="h-4 w-4 mr-2" />
                Copy JSON
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleDownload(
                  formatJson(displayData),
                  `scraped-data-${Date.now()}.json`
                )}
              >
                <Download className="h-4 w-4 mr-2" />
                Download
              </Button>
            </div>
          </div>
        </CardHeader>
        
        {data.metadata && (
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {data.metadata.processingTime && (
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {(data.metadata.processingTime / 1000).toFixed(1)}s
                  </div>
                  <div className="text-sm text-muted-foreground">Processing Time</div>
                </div>
              )}
              {data.metadata.qualityScore && (
                <div className="text-center">
                  <div className={`text-2xl font-bold ${getQualityColor(data.metadata.qualityScore)}`}>
                    {getQualityLabel(data.metadata.qualityScore)}
                  </div>
                  <div className="text-sm text-muted-foreground">Quality Score</div>
                </div>
              )}
              {data.metadata.strategy && (
                <div className="text-center">
                  <Badge variant="secondary" className="text-sm">
                    {data.metadata.strategy}
                  </Badge>
                  <div className="text-sm text-muted-foreground mt-1">Strategy</div>
                </div>
              )}
              {data.metadata.contentType && (
                <div className="text-center">
                  <Badge variant="outline" className="text-sm">
                    {data.metadata.contentType}
                  </Badge>
                  <div className="text-sm text-muted-foreground mt-1">Content Type</div>
                </div>
              )}
            </div>
          </CardContent>
        )}
      </Card>

      {/* Results Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="extracted" className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Extracted Data
          </TabsTrigger>
          <TabsTrigger value="images" className="flex items-center gap-2">
            <Image className="h-4 w-4" />
            Images ({data.images?.length || 0})
          </TabsTrigger>
          <TabsTrigger value="tables" className="flex items-center gap-2">
            <TableIcon className="h-4 w-4" />
            Tables ({data.tables?.length || 0})
          </TabsTrigger>
          <TabsTrigger value="raw" className="flex items-center gap-2">
            <Code className="h-4 w-4" />
            Raw Data
          </TabsTrigger>
        </TabsList>

        <TabsContent value="extracted" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Extracted Content</CardTitle>
              <CardDescription>
                Structured data extracted using AI-powered analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isTableData ? (
                <ScrollArea className="h-96 w-full">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        {Object.keys(displayData[0]).map((header) => (
                          <TableHead key={header}>{header}</TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {displayData.map((row: any, rowIndex: number) => (
                        <TableRow key={rowIndex}>
                          {Object.keys(displayData[0]).map((header) => (
                            <TableCell key={`${rowIndex}-${header}`}>
                              {typeof row[header] === 'object' ? JSON.stringify(row[header]) : String(row[header])}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </ScrollArea>
              ) : (
                <ScrollArea className="h-96 w-full">
                  <pre className="text-sm bg-muted p-4 rounded-lg overflow-auto">
                    {formatJson(displayData)}
                  </pre>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="images" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Extracted Images</CardTitle>
              <CardDescription>
                Images found and analyzed during the scraping process
              </CardDescription>
            </CardHeader>
            <CardContent>
              {data.images && data.images.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {data.images.map((image: any, index: number) => (
                    <Card key={index} className="overflow-hidden">
                      <div className="aspect-video bg-muted flex items-center justify-center">
                        <img
                          src={image.url}
                          alt={image.alt || `Image ${index + 1}`}
                          className="max-w-full max-h-full object-contain"
                          onError={(e) => {
                            (e.target as HTMLImageElement).style.display = 'none';
                          }}
                        />
                      </div>
                      <CardContent className="p-3">
                        <p className="text-sm font-medium truncate">{image.alt || 'Untitled'}</p>
                        {image.description && (
                          <p className="text-xs text-muted-foreground mt-1">
                            {image.description}
                          </p>
                        )}
                        <Button
                          variant="outline"
                          size="sm"
                          className="w-full mt-2"
                          onClick={() => window.open(image.url, '_blank')}
                        >
                          <Eye className="h-3 w-3 mr-1" />
                          View
                        </Button>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Image className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No images were extracted from this page</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tables" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Extracted Tables</CardTitle>
              <CardDescription>
                Structured table data found on the page
              </CardDescription>
            </CardHeader>
            <CardContent>
              {data.tables && data.tables.length > 0 ? (
                <div className="space-y-6">
                  {data.tables.map((table: any, index: number) => (
                    <div key={index} className="border rounded-lg overflow-hidden">
                      {table.caption && (
                        <div className="bg-muted px-4 py-2 border-b">
                          <h4 className="font-medium">{table.caption}</h4>
                        </div>
                      )}
                      <ScrollArea className="w-full">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              {table.headers.map((header: string, headerIndex: number) => (
                                <TableHead key={headerIndex}>
                                  {header}
                                </TableHead>
                              ))}
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {table.rows.map((row: string[], rowIndex: number) => (
                              <TableRow key={rowIndex}>
                                {row.map((cell: string, cellIndex: number) => (
                                  <TableCell key={cellIndex}>
                                    {cell}
                                  </TableCell>
                                ))}
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </ScrollArea>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <TableIcon className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No tables were found on this page</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="raw" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Raw Data</CardTitle>
              <CardDescription>
                Complete raw response and metadata from the scraping process
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="json">
                <TabsList>
                  <TabsTrigger value="json">JSON Response</TabsTrigger>
                  {data.rawHtml && <TabsTrigger value="html">Raw HTML</TabsTrigger>}
                  {data.structuredData && <TabsTrigger value="structured">Structured Data</TabsTrigger>}
                </TabsList>
                
                <TabsContent value="json">
                  <ScrollArea className="h-96 w-full">
                    <pre className="text-sm bg-muted p-4 rounded-lg overflow-auto">
                      {formatJson(data)}
                    </pre>
                  </ScrollArea>
                </TabsContent>
                
                {data.rawHtml && (
                  <TabsContent value="html">
                    <ScrollArea className="h-96 w-full">
                      <pre className="text-sm bg-muted p-4 rounded-lg overflow-auto">
                        {data.rawHtml}
                      </pre>
                    </ScrollArea>
                  </TabsContent>
                )}
                
                {data.structuredData && (
                  <TabsContent value="structured">
                    <ScrollArea className="h-96 w-full">
                      <pre className="text-sm bg-muted p-4 rounded-lg overflow-auto">
                        {formatJson(data.structuredData)}
                      </pre>
                    </ScrollArea>
                  </TabsContent>
                )}
              </Tabs>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
