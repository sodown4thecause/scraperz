"use client";

import { useState } from "react";
import { useAuth } from "@clerk/nextjs";
import Header from "@/components/Header";
import EnhancedDashboard from "@/components/EnhancedDashboard";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loading } from "@/components/ui/loading";
import { useToast } from "@/hooks/use-toast";
import ResultsDisplay from "@/components/ResultsDisplay";
import { Play, Settings, History, BarChart3 } from "lucide-react";
import ScrapingAnimation from "@/components/ScrapingAnimation";

export default function Home() {
  const { isLoaded, userId } = useAuth();
  const [url, setUrl] = useState("");
  const [customPrompt, setCustomPrompt] = useState("");
  const [useMultiModal, setUseMultiModal] = useState(false);
  const [useIncremental, setUseIncremental] = useState(false);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const { toast } = useToast();

  if (!isLoaded) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <div className="flex items-center justify-center h-96">
          <Loading overlay text="Loading application..." />
        </div>
      </div>
    );
  }

  if (!userId) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <div className="flex items-center justify-center h-96">
          <Card className="w-full max-w-md">
            <CardHeader className="text-center">
              <CardTitle className="text-2xl font-bold">
                Welcome to Scraperz
              </CardTitle>
              <CardDescription>
                Please sign in to access the intelligent scraping platform
              </CardDescription>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-muted-foreground mb-4">
                Advanced AI-powered web scraping with real-time monitoring
              </p>
              <div className="flex justify-center space-x-2">
                <Badge variant="secondary">Multi-modal Extraction</Badge>
                <Badge variant="secondary">Adaptive Strategies</Badge>
                <Badge variant="secondary">Real-time Analytics</Badge>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userId) {
      setError("You must be signed in to start a scraping job.");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);

    try {
      const response = await fetch("/api/scrape", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          url, 
          prompt: customPrompt, 
          userId,
          useMultiModal,
          useIncremental
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Something went wrong");
      }

      const result = await response.json();
      setResults(result.data);
      toast({
        title: "Scraping completed",
        description: "Your data has been successfully extracted.",
      });
    } catch (err: any) {
      setError(err.message);
      toast({
        title: "Error",
        description: err.message,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold tracking-tight mb-2">Intelligent Web Scraper</h1>
          <p className="text-xl text-muted-foreground">
            Advanced AI-powered data extraction with real-time monitoring
          </p>
        </div>

        <Tabs defaultValue="scraper" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="scraper" className="flex items-center gap-2">
              <Play className="h-4 w-4" />
              Scraper
            </TabsTrigger>
            <TabsTrigger value="dashboard" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Dashboard
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center gap-2">
              <History className="h-4 w-4" />
              History
            </TabsTrigger>
            <TabsTrigger value="settings" className="flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Settings
            </TabsTrigger>
          </TabsList>

          <TabsContent value="scraper" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Create New Scraping Job</CardTitle>
                <CardDescription>
                  Configure your intelligent scraping parameters with advanced AI extraction
                </CardDescription>
              </CardHeader>
              <form onSubmit={handleSubmit}>
                <CardContent className="space-y-6">
                  <div className="space-y-2">
                    <Label htmlFor="url">Target URL</Label>
                    <Input
                      id="url"
                      placeholder="https://example.com/data-page"
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="prompt">Extraction Prompt</Label>
                    <Textarea
                      id="prompt"
                      placeholder="Extract all product names, prices, and descriptions from this e-commerce page"
                      value={customPrompt}
                      onChange={(e) => setCustomPrompt(e.target.value)}
                      required
                      rows={3}
                    />
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-center space-x-2">
                      <Switch 
                        id="multi-modal" 
                        checked={useMultiModal} 
                        onCheckedChange={setUseMultiModal} 
                      />
                      <Label htmlFor="multi-modal">Multi-modal Extraction</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Switch 
                        id="incremental" 
                        checked={useIncremental} 
                        onCheckedChange={setUseIncremental} 
                      />
                      <Label htmlFor="incremental">Incremental Updates</Label>
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button type="submit" disabled={loading || !userId} className="w-full">
                    {loading ? (
                      <>
                        <Loading className="mr-2" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Start Intelligent Scrape
                      </>
                    )}
                  </Button>
                </CardFooter>
              </form>
            </Card>

            {error && (
              <Card className="border-destructive">
                <CardHeader>
                  <CardTitle className="text-destructive">Scraping Error</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-destructive">{error}</p>
                </CardContent>
              </Card>
            )}

            {results && <ResultsDisplay data={results} />}
          </TabsContent>

          <TabsContent value="dashboard">
            <EnhancedDashboard />
          </TabsContent>

          <TabsContent value="history">
            <Card>
              <CardHeader>
                <CardTitle>Scraping History</CardTitle>
                <CardDescription>
                  View and manage your previous scraping jobs
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">History feature coming soon...</p>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="settings">
            <Card>
              <CardHeader>
                <CardTitle>Scraper Settings</CardTitle>
                <CardDescription>
                  Configure your scraping preferences and API settings
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">Settings panel coming soon...</p>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

        </Tabs>
      </main>
    </div>
  );
}
