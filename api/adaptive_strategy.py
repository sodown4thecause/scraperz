"""
Adaptive Strategy Engine
Intelligently selects and orchestrates scraping tools based on content analysis.
"""

import os
import json
import asyncio
import time
import random
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import requests
from urllib.parse import urlparse, urljoin
import numpy as np
from collections import defaultdict

from intelligent_analyzer import IntelligentAnalyzer, ContentAnalysis
from scrapegraphai.graphs import SmartScraperGraph
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from crawl4ai import AsyncWebCrawler
from multimodal_extractor import MultiModalExtractor
from anti_detection_engine import AdvancedAntiDetectionEngine
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scraper_project.spiders.intelligent_spider import IntelligentSpider

logger = logging.getLogger(__name__)

@dataclass
class StrategyResult:
    """Result from a scraping strategy execution"""
    strategy: str
    success: bool
    data: Any
    execution_time: float
    confidence_score: float
    error_message: Optional[str] = None
    anti_detection_triggered: bool = False
    retry_count: int = 0
    data_quality_score: float = 0.0

@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy"""
    strategy: str
    content_type: str
    success_rate: float
    avg_execution_time: float
    avg_confidence: float
    usage_count: int
    last_used: datetime = field(default_factory=datetime.now)
    domain_success_rate: Dict[str, float] = field(default_factory=dict)

@dataclass
class AntiDetectionState:
    """State tracking for anti-detection measures"""
    domain: str
    last_request_time: datetime
    request_count: int = 0
    detection_events: List[str] = field(default_factory=list)
    current_user_agent: str = ""
    proxy_rotation_index: int = 0
    rate_limit_delay: float = 1.0

class AdaptiveStrategyEngine:
    """Engine that adapts scraping strategy based on content analysis and performance"""

    def __init__(self, anti_detection_engine: AdvancedAntiDetectionEngine):
        self.analyzer = IntelligentAnalyzer()
        self.multimodal_extractor = MultiModalExtractor()
        self.anti_detection_engine = anti_detection_engine
        self.performance_history: Dict[str, List[StrategyPerformance]] = {}
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.anti_detection_states: Dict[str, AntiDetectionState] = {}
        self.proxy_pool: List[str] = self._load_proxy_pool()
        self.user_agents: List[str] = self._load_user_agents()
        self.active_sessions: Dict[str, str] = {}  # domain -> session_id mapping
        
        # Strategy configurations with enhanced capabilities
        self.strategy_configs = {
            'scrapegraph': {
                'name': 'ScrapeGraphAI',
                'description': 'Semantic understanding with LLM',
                'best_for': ['article', 'product', 'review', 'structured_data'],
                'timeout': 60,
                'anti_detection_level': 'medium',
                'resource_intensity': 'high'
            },
            'crawl4ai': {
                'name': 'Crawl4AI',
                'description': 'Dynamic content extraction',
                'best_for': ['spa', 'javascript_heavy', 'dynamic_content'],
                'timeout': 45,
                'anti_detection_level': 'high',
                'resource_intensity': 'medium'
            },
            'scrapy': {
                'name': 'Scrapy',
                'description': 'Distributed crawling',
                'best_for': ['large_sites', 'navigation_heavy', 'batch_processing'],
                'timeout': 120,
                'anti_detection_level': 'low',
                'resource_intensity': 'low'
            },
            'hybrid': {
                'name': 'Hybrid Strategy',
                'description': 'Combines multiple engines intelligently',
                'best_for': ['complex_sites', 'high_security', 'multi_modal'],
                'timeout': 180,
                'anti_detection_level': 'maximum',
                'resource_intensity': 'high'
            }
        }

    async def execute_adaptive_scraping(
        self,
        url: str,
        prompt: str,
        user_id: str,
        force_strategy: Optional[str] = None
    ) -> StrategyResult:
        """
        Execute adaptive scraping with intelligent strategy selection and advanced anti-detection
        """
        start_time = time.time()
        domain = urlparse(url).netloc

        try:
            # Create or get anti-detection session
            session_id = await self._get_or_create_session(domain, force_strategy)
            
            # Analyze the page first
            analysis = await self.analyzer.analyze_page(url)

            # Determine best strategy
            if force_strategy and force_strategy in self.strategy_configs:
                strategy = force_strategy
            else:
                strategy = self._select_optimal_strategy(analysis, url)

            logger.info(f"Selected strategy '{strategy}' for {url} with session {session_id} (content_type: {analysis.content_type})")

            # Get advanced anti-detection configuration
            session_config = await self.anti_detection_engine.get_session_config(session_id)
            
            # Execute the strategy with anti-detection measures
            result = await self._execute_strategy_with_anti_detection(
                strategy, url, prompt, analysis, session_config
            )

            # Detect anti-scraping measures in response
            if result.success and result.data:
                detection_result = await self.anti_detection_engine.detect_anti_scraping_measures(
                    str(result.data), 200, {}
                )
                
                if detection_result['detected']:
                    logger.warning(f"Anti-scraping measures detected: {detection_result['measures']}")
                    result.anti_detection_triggered = True
                    
                    # Apply recommended actions
                    for action in detection_result['recommended_actions']:
                        if action == 'rotate_fingerprint':
                            await self.anti_detection_engine.rotate_session_fingerprint(session_id)
                        elif action == 'change_proxy':
                            # Create new session with different proxy
                            session_id = await self.anti_detection_engine.create_session(domain)
                            self.active_sessions[domain] = session_id

            execution_time = time.time() - start_time

            # Update performance metrics
            self._update_performance_metrics(strategy, analysis.content_type, result.success, execution_time, result.confidence_score, domain)
            
            # Update proxy performance if used
            if session_config.get('proxy'):
                await self.anti_detection_engine.update_proxy_performance(
                    session_config['proxy']['url'],
                    result.success,
                    execution_time
                )

            return StrategyResult(
                strategy=strategy,
                success=result.success,
                data=result.data,
                execution_time=execution_time,
                confidence_score=result.confidence_score,
                error_message=result.error_message if not result.success else None,
                anti_detection_triggered=result.anti_detection_triggered
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in adaptive scraping: {str(e)}")
            
            # Record anti-detection event if relevant
            if "blocked" in str(e).lower() or "captcha" in str(e).lower() or "403" in str(e):
                self._record_anti_detection_event(domain, str(e))
            
            return StrategyResult(
                strategy="error",
                success=False,
                data=None,
                execution_time=execution_time,
                confidence_score=0.0,
                error_message=str(e),
                anti_detection_triggered="blocked" in str(e).lower() or "captcha" in str(e).lower()
            )

    def _select_optimal_strategy(self, analysis: ContentAnalysis, url: str = None) -> str:
        """Select the optimal strategy based on analysis, performance history, and anti-detection needs"""
        domain = urlparse(url).netloc if url else "unknown"
        content_type = analysis.content_type
        
        # Check for anti-detection requirements
        anti_detection_needed = self._assess_anti_detection_needs(domain, analysis)
        
        # Get performance history for this content type
        if content_type in self.performance_history:
            performances = self.performance_history[content_type]
            
            # Filter strategies based on anti-detection needs
            if anti_detection_needed:
                performances = [p for p in performances if 
                              self.strategy_configs.get(p.strategy, {}).get('anti_detection_level') in ['high', 'maximum']]
            
            if performances:
                # Calculate weighted score considering multiple factors
                best_performance = max(performances, key=lambda p: self._calculate_strategy_score(p, domain, analysis))
                return best_performance.strategy
        
        # Enhanced rule-based selection with anti-detection consideration
        if anti_detection_needed:
            if analysis.javascript_heavy or analysis.content_type == 'spa':
                return 'hybrid'  # Use hybrid for complex anti-detection scenarios
            else:
                return 'crawl4ai'  # High anti-detection capability
        
        # Fallback to enhanced analysis-based selection
        return self._get_strategy_from_analysis(analysis)

    async def _execute_strategy_with_anti_detection(
        self,
        strategy: str,
        url: str,
        prompt: str,
        analysis: ContentAnalysis,
        session_config: Dict[str, Any]
    ) -> StrategyResult:
        """Execute strategy with anti-detection configuration"""
        try:
            if strategy == 'scrapegraph':
                return await self._execute_scrapegraph_with_config(url, prompt, analysis, session_config)
            elif strategy == 'crawl4ai':
                return await self._execute_crawl4ai_with_config(url, prompt, analysis, session_config)
            elif strategy == 'scrapy':
                return await self._execute_scrapy_with_config(url, prompt, analysis, session_config)
            elif strategy == 'hybrid':
                return await self._execute_hybrid_strategy_with_config(url, prompt, analysis, session_config)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        except Exception as e:
            logger.error(f"Error executing strategy {strategy} with anti-detection: {str(e)}")
            return StrategyResult(
                strategy=strategy,
                success=False,
                data=None,
                execution_time=0.0,
                confidence_score=0.0,
                error_message=str(e),
                anti_detection_triggered="blocked" in str(e).lower() or "captcha" in str(e).lower()
            )

    async def _execute_scrapegraph(
        self,
        url: str,
        prompt: str,
        analysis: ContentAnalysis
    ) -> StrategyResult:
        """Execute ScrapeGraphAI strategy"""
        start_time = time.time()

        try:
            # Configure ScrapeGraphAI
            graph_config = {
                "llm": {
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    "model": "gemini-pro",
                },
                "verbose": False,
                "headless": True,
            }

            # Create and run the scraper
            smart_scraper_graph = SmartScraperGraph(
                prompt=prompt,
                source=url,
                config=graph_config
            )

            result = smart_scraper_graph.run()
            execution_time = time.time() - start_time

            # Calculate confidence based on result quality
            confidence = self._calculate_result_confidence(result)

            return StrategyResult(
                strategy="scrapegraph",
                success=True,
                data=result,
                execution_time=execution_time,
                confidence_score=confidence
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return StrategyResult(
                strategy="scrapegraph",
                success=False,
                data=None,
                execution_time=execution_time,
                confidence_score=0.0,
                error_message=str(e)
            )

    async def _execute_crawl4ai(
        self,
        url: str,
        prompt: str,
        analysis: ContentAnalysis
    ) -> StrategyResult:
        """Execute Crawl4AI strategy"""
        start_time = time.time()

        try:
            # Use Crawl4AI to get rendered HTML
            crawler = Crawl4AI(url=url)
            rendered_html = await crawler.fetch_html()

            if not rendered_html:
                raise Exception("Crawl4AI failed to fetch content")

            # Use ScrapeGraphAI on the rendered HTML
            graph_config = {
                "llm": {
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    "model": "gemini-pro",
                },
                "verbose": False,
                "headless": True,
            }

            smart_scraper_graph = SmartScraperGraph(
                prompt=prompt,
                source=rendered_html,
                config=graph_config
            )

            result = smart_scraper_graph.run()
            execution_time = time.time() - start_time

            confidence = self._calculate_result_confidence(result)

            return StrategyResult(
                strategy="crawl4ai",
                success=True,
                data=result,
                execution_time=execution_time,
                confidence_score=confidence
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return StrategyResult(
                strategy="crawl4ai",
                success=False,
                data=None,
                execution_time=execution_time,
                confidence_score=0.0,
                error_message=str(e)
            )

    async def _execute_scrapy(
        self,
        url: str,
        prompt: str,
        analysis: ContentAnalysis
    ) -> StrategyResult:
        """Execute Scrapy strategy"""
        start_time = time.time()

        try:
            # Use multiprocessing to run Scrapy
            manager = multiprocessing.Manager()
            results_queue = manager.Queue()

            # Prepare spider arguments
            graph_config = {
                "llm": {
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    "model": "gemini-pro",
                },
                "verbose": False,
                "headless": True,
            }

            spider_kwargs = {
                'start_url': url,
                'prompt': prompt,
                'llm_config': graph_config,
                'results_list': manager.list()
            }

            # Run spider in separate process
            process = multiprocessing.Process(
                target=self._run_scrapy_spider,
                args=(IntelligentSpider, spider_kwargs, results_queue)
            )
            process.start()
            process.join(timeout=120)  # 2 minute timeout

            if process.is_alive():
                process.terminate()
                raise Exception("Scrapy spider timed out")

            # Get results
            if not results_queue.empty():
                results = results_queue.get()
                if isinstance(results, Exception):
                    raise results

                execution_time = time.time() - start_time
                confidence = self._calculate_result_confidence(results)

                return StrategyResult(
                    strategy="scrapy",
                    success=True,
                    data=results,
                    execution_time=execution_time,
                    confidence_score=confidence
                )
            else:
                raise Exception("No results from Scrapy spider")

        except Exception as e:
            execution_time = time.time() - start_time
            return StrategyResult(
                strategy="scrapy",
                success=False,
                data=None,
                execution_time=execution_time,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _run_scrapy_spider(self, spider_class, kwargs, queue):
        """Run Scrapy spider in a separate process"""
        try:
            results = []
            process = CrawlerProcess(get_project_settings())

            spider_kwargs = dict(kwargs)
            spider_kwargs['results_list'] = results

            process.crawl(spider_class, **spider_kwargs)
            process.start()

            queue.put(results)
        except Exception as e:
            queue.put(e)

    def _calculate_result_confidence(self, result: Any) -> float:
        """Calculate confidence score for scraping results"""
        if not result:
            return 0.0

        try:
            # Convert result to string for analysis
            result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)

            # Calculate confidence based on result characteristics
            confidence = 0.5  # Base confidence

            # Increase confidence for structured data
            if isinstance(result, (dict, list)):
                confidence += 0.2

            # Increase confidence for substantial content
            if len(result_str) > 100:
                confidence += 0.2

            # Increase confidence for multiple data points
            if isinstance(result, list) and len(result) > 1:
                confidence += 0.1

            return min(confidence, 1.0)

        except Exception:
            return 0.5

    def _update_performance_metrics(
        self,
        strategy: str,
        content_type: str,
        success: bool,
        execution_time: float,
        confidence: float,
        domain: str = None
    ):
        """Update performance metrics for strategy optimization"""
        if content_type not in self.performance_history:
            self.performance_history[content_type] = []

        performances = self.performance_history[content_type]

        # Find existing performance record
        existing_perf = None
        for perf in performances:
            if perf.strategy == strategy:
                existing_perf = perf
                break

        if existing_perf:
            # Update existing record
            total_executions = existing_perf.usage_count + 1
            existing_perf.success_rate = ((existing_perf.success_rate * existing_perf.usage_count) + (1 if success else 0)) / total_executions
            existing_perf.avg_execution_time = ((existing_perf.avg_execution_time * existing_perf.usage_count) + execution_time) / total_executions
            existing_perf.avg_confidence = ((existing_perf.avg_confidence * existing_perf.usage_count) + confidence) / total_executions
            existing_perf.usage_count = total_executions
            existing_perf.last_used = datetime.now()
            
            # Update domain-specific success rate
            if domain:
                if domain not in existing_perf.domain_success_rate:
                    existing_perf.domain_success_rate[domain] = 1.0 if success else 0.0
                else:
                    # Simple moving average for domain success
                    existing_perf.domain_success_rate[domain] = (existing_perf.domain_success_rate[domain] * 0.8) + ((1.0 if success else 0.0) * 0.2)
        else:
            # Create new performance record
            new_perf = StrategyPerformance(
                strategy=strategy,
                content_type=content_type,
                success_rate=1.0 if success else 0.0,
                avg_execution_time=execution_time,
                avg_confidence=confidence,
                usage_count=1,
                last_used=datetime.now(),
                domain_success_rate={domain: 1.0 if success else 0.0} if domain else {}
            )
            performances.append(new_perf)

    def get_strategy_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report for all strategies"""
        report = {}

        for content_type, performances in self.performance_history.items():
            report[content_type] = []
            for perf in performances:
                report[content_type].append({
                    'strategy': perf.strategy,
                    'success_rate': perf.success_rate,
                    'avg_execution_time': perf.avg_execution_time,
                    'avg_confidence': perf.avg_confidence,
                    'usage_count': perf.usage_count
                })

        return report

    async def analyze_page_only(self, url: str) -> ContentAnalysis:
        """Analyze a page without executing scraping"""
        return await self.analyzer.analyze_page(url)
    
    def _load_proxy_pool(self) -> List[str]:
        """Load proxy pool from environment or configuration"""
        proxy_list = os.getenv('PROXY_LIST', '').split(',')
        return [proxy.strip() for proxy in proxy_list if proxy.strip()]
    
    def _load_user_agents(self) -> List[str]:
        """Load realistic user agent strings"""
        return [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
    
    def _assess_anti_detection_needs(self, domain: str, analysis: ContentAnalysis) -> bool:
        """Assess if anti-detection measures are needed for this domain"""
        # Check domain history for detection events
        if domain in self.anti_detection_states:
            state = self.anti_detection_states[domain]
            if len(state.detection_events) > 0:
                return True
        
        # Check for high-security indicators in analysis
        security_indicators = [
            'cloudflare' in analysis.technologies,
            'recaptcha' in analysis.technologies,
            analysis.content_type in ['e_commerce', 'financial', 'social_media'],
            analysis.complexity_score > 0.8
        ]
        
        return any(security_indicators)
    
    def _calculate_strategy_score(self, performance: StrategyPerformance, domain: str, analysis: ContentAnalysis) -> float:
        """Calculate weighted score for strategy selection"""
        base_score = performance.success_rate * performance.avg_confidence
        
        # Penalize slow strategies
        time_penalty = min(performance.avg_execution_time / 60.0, 1.0)  # Normalize to 1 minute
        base_score *= (1.0 - time_penalty * 0.3)
        
        # Boost score for domain-specific success
        if domain in performance.domain_success_rate:
            domain_boost = performance.domain_success_rate[domain] * 0.2
            base_score += domain_boost
        
        # Consider recency of usage
        days_since_use = (datetime.now() - performance.last_used).days
        recency_factor = max(0.5, 1.0 - (days_since_use / 30.0))  # Decay over 30 days
        base_score *= recency_factor
        
        return base_score
    
    def _get_strategy_from_analysis(self, analysis: ContentAnalysis) -> str:
        """Get strategy based on enhanced content analysis"""
        # Multi-factor decision making
        if analysis.javascript_heavy and analysis.complexity_score > 0.7:
            return 'crawl4ai'
        elif analysis.content_type in ['article', 'product', 'review'] and analysis.structure_score > 0.6:
            return 'scrapegraph'
        elif analysis.content_type == 'navigation_heavy' or analysis.complexity_score < 0.4:
            return 'scrapy'
        else:
            return analysis.recommended_strategy or 'scrapegraph'
    
    def _record_anti_detection_event(self, domain: str, event_description: str):
        """Record an anti-detection event for learning"""
        if domain not in self.anti_detection_states:
            self.anti_detection_states[domain] = AntiDetectionState(
                domain=domain,
                last_request_time=datetime.now()
            )
        
        state = self.anti_detection_states[domain]
        state.detection_events.append(f"{datetime.now().isoformat()}: {event_description}")
        state.rate_limit_delay = min(state.rate_limit_delay * 1.5, 10.0)  # Increase delay, max 10s
        
        logger.warning(f"Anti-detection event recorded for {domain}: {event_description}")
    
    async def _apply_anti_detection_measures(self, domain: str) -> Dict[str, Any]:
        """Apply anti-detection measures for a domain"""
        if domain not in self.anti_detection_states:
            self.anti_detection_states[domain] = AntiDetectionState(
                domain=domain,
                last_request_time=datetime.now()
            )
        
        state = self.anti_detection_states[domain]
        
        # Rate limiting
        time_since_last = (datetime.now() - state.last_request_time).total_seconds()
        if time_since_last < state.rate_limit_delay:
            await asyncio.sleep(state.rate_limit_delay - time_since_last)
        
        # Rotate user agent
        if not state.current_user_agent or random.random() < 0.1:  # 10% chance to rotate
            state.current_user_agent = random.choice(self.user_agents)
        
        # Proxy rotation
        proxy = None
        if self.proxy_pool and (not state.detection_events or random.random() < 0.3):
            state.proxy_rotation_index = (state.proxy_rotation_index + 1) % len(self.proxy_pool)
            proxy = self.proxy_pool[state.proxy_rotation_index]
        
        state.last_request_time = datetime.now()
        state.request_count += 1
        
        return {
            'user_agent': state.current_user_agent,
            'proxy': proxy,
            'delay': random.uniform(1, 3),  # Random delay between requests
            'headers': self._generate_realistic_headers(state.current_user_agent)
        }
    
    def _generate_realistic_headers(self, user_agent: str) -> Dict[str, str]:
        """Generate realistic HTTP headers"""
        return {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
    
    async def _execute_hybrid_strategy(self, url: str, prompt: str, analysis: ContentAnalysis) -> StrategyResult:
        """Execute hybrid strategy combining multiple engines"""
        start_time = time.time()
        
        try:
            # Apply anti-detection measures
            domain = urlparse(url).netloc
            anti_detection_config = await self._apply_anti_detection_measures(domain)
            
            # Try Crawl4AI first for dynamic content
            crawl4ai_result = await self._execute_crawl4ai_with_config(url, prompt, analysis, anti_detection_config)
            
            if crawl4ai_result.success and crawl4ai_result.confidence_score > 0.7:
                return crawl4ai_result
            
            # Fallback to ScrapeGraphAI for semantic understanding
            scrapegraph_result = await self._execute_scrapegraph_with_config(url, prompt, analysis, anti_detection_config)
            
            # Choose best result
            if scrapegraph_result.success and scrapegraph_result.confidence_score > crawl4ai_result.confidence_score:
                execution_time = time.time() - start_time
                scrapegraph_result.execution_time = execution_time
                scrapegraph_result.strategy = 'hybrid'
                return scrapegraph_result
            
            execution_time = time.time() - start_time
            crawl4ai_result.execution_time = execution_time
            crawl4ai_result.strategy = 'hybrid'
            return crawl4ai_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return StrategyResult(
                strategy='hybrid',
                success=False,
                data=None,
                execution_time=execution_time,
                confidence_score=0.0,
                error_message=str(e),
                anti_detection_triggered=True
            )
    
    async def _get_or_create_session(self, domain: str, force_strategy: Optional[str] = None) -> str:
        """Get existing session or create new one for domain"""
        if domain in self.active_sessions:
            session_id = self.active_sessions[domain]
            # Validate session is still active
            if await self.anti_detection_engine.is_session_active(session_id):
                return session_id
        
        # Create new session
        session_id = await self.anti_detection_engine.create_session(domain)
        self.active_sessions[domain] = session_id
        return session_id
    
    async def _execute_strategy_with_anti_detection(
        self,
        strategy: str,
        url: str,
        prompt: str,
        analysis: ContentAnalysis,
        session_config: Dict[str, Any]
    ) -> StrategyResult:
        """Execute strategy with anti-detection configuration"""
        try:
            if strategy == 'scrapegraph':
                return await self._execute_scrapegraph_with_config(url, prompt, analysis, session_config)
            elif strategy == 'crawl4ai':
                return await self._execute_crawl4ai_with_config(url, prompt, analysis, session_config)
            elif strategy == 'scrapy':
                return await self._execute_scrapy_with_config(url, prompt, analysis, session_config)
            elif strategy == 'hybrid':
                return await self._execute_hybrid_strategy_with_config(url, prompt, analysis, session_config)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        except Exception as e:
            logger.error(f"Error executing strategy {strategy} with anti-detection: {str(e)}")
            return StrategyResult(
                strategy=strategy,
                success=False,
                data=None,
                execution_time=0.0,
                confidence_score=0.0,
                error_message=str(e),
                anti_detection_triggered="blocked" in str(e).lower() or "captcha" in str(e).lower()
            )
    
    async def _execute_hybrid_strategy_with_config(self, url: str, prompt: str, analysis: ContentAnalysis, config: Dict[str, Any]) -> StrategyResult:
        """Execute hybrid strategy with anti-detection configuration"""
        start_time = time.time()
        
        try:
            # Try Crawl4AI first for dynamic content
            crawl4ai_result = await self._execute_crawl4ai_with_config(url, prompt, analysis, config)
            
            if crawl4ai_result.success and crawl4ai_result.confidence_score > 0.7:
                return crawl4ai_result
            
            # Fallback to ScrapeGraphAI for semantic understanding
            scrapegraph_result = await self._execute_scrapegraph_with_config(url, prompt, analysis, config)
            
            # Choose best result
            if scrapegraph_result.success and scrapegraph_result.confidence_score > crawl4ai_result.confidence_score:
                execution_time = time.time() - start_time
                scrapegraph_result.execution_time = execution_time
                scrapegraph_result.strategy = 'hybrid'
                return scrapegraph_result
            
            execution_time = time.time() - start_time
            crawl4ai_result.execution_time = execution_time
            crawl4ai_result.strategy = 'hybrid'
            return crawl4ai_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return StrategyResult(
                strategy='hybrid',
                success=False,
                data=None,
                execution_time=execution_time,
                confidence_score=0.0,
                error_message=str(e),
                anti_detection_triggered=True
            )
    
    async def _execute_scrapy_with_config(self, url: str, prompt: str, analysis: ContentAnalysis, config: Dict[str, Any]) -> StrategyResult:
        """Execute Scrapy strategy with anti-detection configuration"""
        start_time = time.time()
        
        try:
            # Apply human-like delays before execution
            if config.get('behavioral_patterns', {}).get('delay_range'):
                delay_range = config['behavioral_patterns']['delay_range']
                delay = random.uniform(*delay_range)
                await asyncio.sleep(delay)
            
            # Enhanced LLM configuration for Scrapy
            enhanced_llm_config = {
                "api_key": os.getenv("GEMINI_API_KEY"),
                "model": "gemini-pro",
            }
            
            # Prepare anti-detection settings for Scrapy spider
            spider_settings = {
                'USER_AGENT': config.get('user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
                'ROBOTSTXT_OBEY': False,
                'CONCURRENT_REQUESTS': 1,  # Conservative for anti-detection
                'DOWNLOAD_DELAY': random.uniform(1, 3),  # Random delays
                'RANDOMIZE_DOWNLOAD_DELAY': 0.5,
                'AUTOTHROTTLE_ENABLED': True,
                'AUTOTHROTTLE_START_DELAY': 1,
                'AUTOTHROTTLE_MAX_DELAY': 10,
                'AUTOTHROTTLE_TARGET_CONCURRENCY': 1.0,
                'COOKIES_ENABLED': True,
                'TELNETCONSOLE_ENABLED': False
            }
            
            # Apply proxy configuration
            if config.get('proxy'):
                proxy_config = config['proxy']
                spider_settings['DOWNLOADER_MIDDLEWARES'] = {
                    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
                }
                # Proxy will be set per request in the spider
            
            # Apply browser fingerprint settings
            if config.get('browser_fingerprint'):
                fingerprint = config['browser_fingerprint']
                spider_settings.update({
                    'USER_AGENT': fingerprint.get('user_agent', spider_settings['USER_AGENT']),
                    'DEFAULT_REQUEST_HEADERS': {
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': fingerprint.get('accept_language', 'en-US,en;q=0.9'),
                        'Accept-Encoding': 'gzip, deflate, br',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Cache-Control': 'max-age=0'
                    }
                })
            
            # Apply behavioral patterns
            if config.get('behavioral_patterns'):
                patterns = config['behavioral_patterns']
                spider_settings.update({
                    'DOWNLOAD_DELAY': random.uniform(*patterns.get('delay_range', [1, 3])),
                    'RANDOMIZE_DOWNLOAD_DELAY': 0.5,
                    'CONCURRENT_REQUESTS_PER_DOMAIN': 1 if patterns.get('simulate_user') else 2
                })
            
            # Execute Scrapy with enhanced anti-detection
            queue = multiprocessing.Queue()
            
            def run_enhanced_spider():
                try:
                    from scrapy.crawler import CrawlerProcess
                    from scrapy.utils.project import get_project_settings
                    
                    # Get base settings and update with anti-detection settings
                    settings = get_project_settings()
                    settings.update(spider_settings)
                    
                    process = CrawlerProcess(settings)
                    
                    # Enhanced spider arguments
                    spider_kwargs = {
                        'start_url': url,
                        'prompt': prompt,
                        'llm_config': enhanced_llm_config,
                        'anti_detection_config': config,
                        'results_list': []
                    }
                    
                    process.crawl('intelligent_spider', **spider_kwargs)
                    process.start()
                    
                    # Get results from spider
                    results = spider_kwargs['results_list']
                    queue.put(results)
                    
                except Exception as e:
                    queue.put(e)
            
            # Run spider in separate process
            spider_process = multiprocessing.Process(target=run_enhanced_spider)
            spider_process.start()
            spider_process.join(timeout=120)  # 2 minute timeout
            
            if spider_process.is_alive():
                spider_process.terminate()
                spider_process.join()
                raise TimeoutError("Scrapy execution timed out")
            
            # Get results
            try:
                results = queue.get_nowait()
                if isinstance(results, Exception):
                    raise results
            except:
                results = []
            
            if results:
                confidence = self._calculate_result_confidence(results)
                
                # Enhance results with anti-detection metadata
                enhanced_results = {
                    'scraped_data': results,
                    'anti_detection_applied': True,
                    'session_config': {
                        'user_agent': config.get('user_agent', 'default'),
                        'proxy_used': bool(config.get('proxy')),
                        'fingerprint_applied': bool(config.get('browser_fingerprint')),
                        'behavioral_patterns_applied': bool(config.get('behavioral_patterns'))
                    },
                    'extraction_metadata': {
                        'strategy': 'scrapy',
                        'llm_model': 'gemini-pro',
                        'content_type': analysis.content_type,
                        'complexity_score': analysis.complexity_score,
                        'download_delay_used': spider_settings.get('DOWNLOAD_DELAY', 1)
                    }
                }
                
                return StrategyResult(
                    strategy='scrapy',
                    success=True,
                    data=enhanced_results,
                    execution_time=time.time() - start_time,
                    confidence_score=confidence
                )
            else:
                return StrategyResult(
                    strategy='scrapy',
                    success=False,
                    data=None,
                    execution_time=time.time() - start_time,
                    confidence_score=0.0,
                    error_message="Scrapy returned no results"
                )
                
        except Exception as e:
            logger.error(f"Scrapy with anti-detection failed: {str(e)}")
            return StrategyResult(
                strategy='scrapy',
                success=False,
                data=None,
                execution_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=f"Scrapy execution error: {str(e)}"
            )
    
    async def _execute_crawl4ai_with_config(self, url: str, prompt: str, analysis: ContentAnalysis, config: Dict[str, Any]) -> StrategyResult:
        """Execute Crawl4AI strategy with anti-detection configuration"""
        start_time = time.time()
        
        try:
            # Configure Crawl4AI with anti-detection settings
            crawler_config = {
                'headless': True,
                'verbose': False,
                'browser_type': 'chromium'
            }
            
            # Apply anti-detection configurations
            if config.get('user_agent'):
                crawler_config['user_agent'] = config['user_agent']
            
            if config.get('proxy'):
                proxy_config = config['proxy']
                crawler_config['proxy'] = proxy_config['url']
                if proxy_config.get('auth'):
                    crawler_config['proxy_auth'] = proxy_config['auth']
            
            # Apply browser fingerprint settings
            if config.get('browser_fingerprint'):
                fingerprint = config['browser_fingerprint']
                crawler_config.update({
                    'viewport_width': fingerprint.get('viewport_width', 1920),
                    'viewport_height': fingerprint.get('viewport_height', 1080),
                    'user_agent': fingerprint.get('user_agent', crawler_config.get('user_agent')),
                    'accept_language': fingerprint.get('accept_language', 'en-US,en;q=0.9'),
                    'timezone': fingerprint.get('timezone', 'America/New_York')
                })
            
            # Apply behavioral patterns
            behavioral_config = {}
            if config.get('behavioral_patterns'):
                patterns = config['behavioral_patterns']
                behavioral_config = {
                    'wait_for_images': patterns.get('wait_for_images', True),
                    'simulate_user': patterns.get('simulate_user', True),
                    'delay_range': patterns.get('delay_range', [1, 3]),
                    'scroll_pause_time': patterns.get('scroll_pause_time', 2)
                }
            
            async with AsyncWebCrawler(**crawler_config) as crawler:
                # Apply human-like delays
                if behavioral_config.get('delay_range'):
                    delay = random.uniform(*behavioral_config['delay_range'])
                    await asyncio.sleep(delay)
                
                # Execute crawling with enhanced options
                crawl_options = {
                    'word_count_threshold': 10,
                    'extraction_strategy': 'LLMExtractionStrategy',
                    'extraction_strategy_args': {
                        'provider': 'google',
                        'api_key': os.getenv('GEMINI_API_KEY'),
                        'model': 'gemini-pro',
                        'instruction': prompt
                    },
                    'chunking_strategy': 'RegexChunking',
                    'bypass_cache': True
                }
                
                # Add behavioral simulation
                if behavioral_config.get('simulate_user'):
                    crawl_options['simulate_user'] = True
                    crawl_options['override_navigator'] = True
                
                if behavioral_config.get('wait_for_images'):
                    crawl_options['wait_for_images'] = True
                
                result = await crawler.arun(
                    url=url,
                    **crawl_options
                )
                
                if result.success and result.extracted_content:
                    confidence = self._calculate_result_confidence(result.extracted_content)
                    
                    return StrategyResult(
                        strategy='crawl4ai',
                        success=True,
                        data={
                            'extracted_content': result.extracted_content,
                            'markdown': result.markdown,
                            'links': result.links,
                            'images': result.images,
                            'metadata': result.metadata,
                            'anti_detection_applied': True,
                            'session_config': {
                                'user_agent': config.get('user_agent', 'default'),
                                'proxy_used': bool(config.get('proxy')),
                                'fingerprint_applied': bool(config.get('browser_fingerprint'))
                            }
                        },
                        execution_time=time.time() - start_time,
                        confidence_score=confidence
                    )
                else:
                    return StrategyResult(
                        strategy='crawl4ai',
                        success=False,
                        data=None,
                        execution_time=time.time() - start_time,
                        confidence_score=0.0,
                        error_message=f"Crawl4AI failed: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}"
                    )
                    
        except Exception as e:
            logger.error(f"Crawl4AI with anti-detection failed: {str(e)}")
            return StrategyResult(
                strategy='crawl4ai',
                success=False,
                data=None,
                execution_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=f"Crawl4AI execution error: {str(e)}"
            )
    
    async def _execute_scrapegraph_with_config(self, url: str, prompt: str, analysis: ContentAnalysis, config: Dict[str, Any]) -> StrategyResult:
        """Execute ScrapeGraphAI with anti-detection configuration"""
        start_time = time.time()
        
        try:
            # Enhanced graph configuration with anti-detection
            enhanced_graph_config = {
                "llm": {
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    "model": "gemini-pro",
                },
                "verbose": False,
                "headless": True,
                "browser_type": "chromium"
            }
            
            # Apply anti-detection configurations
            if config.get('user_agent'):
                enhanced_graph_config['user_agent'] = config['user_agent']
            
            if config.get('proxy'):
                proxy_config = config['proxy']
                enhanced_graph_config['proxy'] = {
                    'server': proxy_config['url']
                }
                if proxy_config.get('auth'):
                    enhanced_graph_config['proxy']['username'] = proxy_config['auth']['username']
                    enhanced_graph_config['proxy']['password'] = proxy_config['auth']['password']
            
            # Apply browser fingerprint settings
            if config.get('browser_fingerprint'):
                fingerprint = config['browser_fingerprint']
                enhanced_graph_config.update({
                    'viewport': {
                        'width': fingerprint.get('viewport_width', 1920),
                        'height': fingerprint.get('viewport_height', 1080)
                    },
                    'user_agent': fingerprint.get('user_agent', enhanced_graph_config.get('user_agent')),
                    'locale': fingerprint.get('accept_language', 'en-US'),
                    'timezone_id': fingerprint.get('timezone', 'America/New_York')
                })
            
            # Apply behavioral patterns
            if config.get('behavioral_patterns'):
                patterns = config['behavioral_patterns']
                enhanced_graph_config.update({
                    'wait_for_images': patterns.get('wait_for_images', True),
                    'simulate_user_interaction': patterns.get('simulate_user', True),
                    'random_delay': patterns.get('delay_range', [1, 3]),
                    'scroll_behavior': {
                        'enabled': True,
                        'pause_time': patterns.get('scroll_pause_time', 2)
                    }
                })
            
            # Apply human-like delays before execution
            if config.get('behavioral_patterns', {}).get('delay_range'):
                delay_range = config['behavioral_patterns']['delay_range']
                delay = random.uniform(*delay_range)
                await asyncio.sleep(delay)
            
            # Create and execute the smart scraper with enhanced config
            smart_scraper_graph = SmartScraperGraph(
                prompt=prompt,
                source=url,
                config=enhanced_graph_config
            )
            
            result = smart_scraper_graph.run()
            
            if result:
                confidence = self._calculate_result_confidence(result)
                
                # Enhance result with anti-detection metadata
                enhanced_result = {
                    'extracted_data': result,
                    'anti_detection_applied': True,
                    'session_config': {
                        'user_agent': config.get('user_agent', 'default'),
                        'proxy_used': bool(config.get('proxy')),
                        'fingerprint_applied': bool(config.get('browser_fingerprint')),
                        'behavioral_patterns_applied': bool(config.get('behavioral_patterns'))
                    },
                    'extraction_metadata': {
                        'strategy': 'scrapegraph',
                        'llm_model': 'gemini-pro',
                        'content_type': analysis.content_type,
                        'complexity_score': analysis.complexity_score
                    }
                }
                
                return StrategyResult(
                    strategy='scrapegraph',
                    success=True,
                    data=enhanced_result,
                    execution_time=time.time() - start_time,
                    confidence_score=confidence
                )
            else:
                return StrategyResult(
                    strategy='scrapegraph',
                    success=False,
                    data=None,
                    execution_time=time.time() - start_time,
                    confidence_score=0.0,
                    error_message="ScrapeGraph returned empty result"
                )
                
        except Exception as e:
            logger.error(f"ScrapeGraph with anti-detection failed: {str(e)}")
            return StrategyResult(
                strategy='scrapegraph',
                success=False,
                data=None,
                execution_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=f"ScrapeGraph execution error: {str(e)}"
            )