"""
Intelligent Content Recognition System
Uses Gemini embeddings to analyze web pages and determine optimal scraping strategies.
"""

import os
import json
import hashlib
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import google.generativeai as genai
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urljoin
import re
from crawl4ai import AsyncWebCrawler
import logging
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContentAnalysis:
    """Analysis result for a web page"""
    content_type: str
    confidence: float
    embedding: List[float]
    fingerprint: str
    structure_score: float
    dynamic_content_ratio: float
    recommended_strategy: str
    features: Dict[str, Any]
    quality_score: float = 0.0
    extraction_patterns: List[str] = None
    semantic_clusters: List[str] = None
    content_freshness: float = 0.0
    
    def __post_init__(self):
        if self.extraction_patterns is None:
            self.extraction_patterns = []
        if self.semantic_clusters is None:
            self.semantic_clusters = []

@dataclass
class PageStructure:
    """Structural analysis of a web page"""
    has_tables: bool
    has_forms: bool
    has_javascript: bool
    has_images: bool
    navigation_depth: int
    content_blocks: int
    text_to_html_ratio: float

class IntelligentAnalyzer:
    """Core intelligence engine for web scraping analysis with learning capabilities"""

    def __init__(self):
        # Initialize Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
        self.embedding_model = genai.GenerativeModel('embedding-001')
        
        # Learning system components
        self.knowledge_base_path = Path("knowledge_base")
        self.knowledge_base_path.mkdir(exist_ok=True)
        self.semantic_memory = self._load_semantic_memory()
        self.pattern_library = self._load_pattern_library()
        self.success_metrics = defaultdict(list)
        
        # Enhanced content type patterns with semantic understanding
        self.content_patterns = {
            'article': {
                'keywords': [
                    r'article', r'post', r'blog', r'news', r'story',
                    r'content', r'publication', r'editorial', r'journalism'
                ],
                'semantic_indicators': [
                    'byline', 'publish date', 'author', 'headline',
                    'paragraph structure', 'reading time'
                ]
            },
            'product': {
                'keywords': [
                    r'product', r'item', r'shop', r'store', r'catalog',
                    r'price', r'buy', r'purchase', r'cart', r'sku'
                ],
                'semantic_indicators': [
                    'price display', 'add to cart', 'product images',
                    'specifications', 'reviews', 'availability'
                ]
            },
            'review': {
                'keywords': [
                    r'review', r'rating', r'comment', r'feedback',
                    r'testimonial', r'opinion', r'score', r'stars'
                ],
                'semantic_indicators': [
                    'star rating', 'user avatar', 'review date',
                    'helpful votes', 'verified purchase'
                ]
            },
            'directory': {
                'keywords': [
                    r'directory', r'listing', r'index', r'category',
                    r'section', r'menu', r'navigation', r'browse'
                ],
                'semantic_indicators': [
                    'category links', 'pagination', 'filter options',
                    'sort controls', 'breadcrumbs'
                ]
            },
            'profile': {
                'keywords': [
                    r'profile', r'user', r'member', r'account',
                    r'person', r'contact', r'bio', r'about'
                ],
                'semantic_indicators': [
                    'profile picture', 'contact info', 'social links',
                    'biography', 'achievements', 'connections'
                ]
            },
            'documentation': {
                'keywords': [
                    r'doc', r'guide', r'tutorial', r'manual',
                    r'help', r'support', r'faq', r'api'
                ],
                'semantic_indicators': [
                    'code examples', 'step by step', 'table of contents',
                    'search functionality', 'version info'
                ]
            }
        }

    async def analyze_page(self, url: str, html_content: Optional[str] = None) -> ContentAnalysis:
        """
        Perform comprehensive analysis of a web page
        """
        try:
            # Get HTML content if not provided
            if not html_content:
                html_content = await self._fetch_page_content(url)

            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract text content
            text_content = self._extract_text_content(soup)

            # Analyze page structure
            structure = self._analyze_structure(soup, html_content)

            # Generate semantic embedding
            embedding = await self._generate_embedding(text_content)

            # Enhanced content classification with semantic understanding
            content_type, confidence = await self._classify_content_enhanced(text_content, url, soup, embedding)

            # Generate semantic fingerprint
            fingerprint = self._generate_semantic_fingerprint(url, text_content, embedding)

            # Determine recommended strategy with learning
            recommended_strategy = self._determine_strategy_enhanced(structure, content_type, confidence, url)

            # Calculate structure score
            structure_score = self._calculate_structure_score(structure)

            # Calculate dynamic content ratio
            dynamic_ratio = self._calculate_dynamic_ratio(html_content)

            # Extract additional features with semantic analysis
            features = self._extract_features_enhanced(soup, url, embedding)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(structure, confidence, features)
            
            # Extract patterns for learning
            extraction_patterns = self._extract_patterns(soup, content_type)
            
            # Find semantic clusters
            semantic_clusters = await self._find_semantic_clusters(embedding, content_type)
            
            # Calculate content freshness
            content_freshness = self._calculate_content_freshness(soup, features)

            return ContentAnalysis(
                content_type=content_type,
                confidence=confidence,
                embedding=embedding,
                fingerprint=fingerprint,
                structure_score=structure_score,
                dynamic_content_ratio=dynamic_ratio,
                recommended_strategy=recommended_strategy,
                features=features,
                quality_score=quality_score,
                extraction_patterns=extraction_patterns,
                semantic_clusters=semantic_clusters,
                content_freshness=content_freshness
            )

        except Exception as e:
            logger.error(f"Error analyzing page {url}: {str(e)}")
            # Return fallback analysis
            return ContentAnalysis(
                content_type="unknown",
                confidence=0.0,
                embedding=[],
                fingerprint=self._generate_semantic_fingerprint(url, "", []),
                structure_score=0.0,
                dynamic_content_ratio=0.0,
                recommended_strategy="scrapegraph",
                features={},
                quality_score=0.0,
                extraction_patterns=[],
                semantic_clusters=[],
                content_freshness=0.0
            )

    async def _fetch_page_content(self, url: str) -> str:
        """Fetch page content with fallback to AsyncWebCrawler for dynamic content"""
        try:
            # Try basic requests first
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Basic fetch failed for {url}, trying AsyncWebCrawler: {str(e)}")
            # Fallback to AsyncWebCrawler for dynamic content
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                return result.html if result else ""

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract meaningful text content from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text content
        text = soup.get_text(separator=' ', strip=True)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text[:10000]  # Limit text length for embedding

    def _analyze_structure(self, soup: BeautifulSoup, html_content: str) -> PageStructure:
        """Analyze the structural characteristics of the page"""
        has_tables = len(soup.find_all('table')) > 0
        has_forms = len(soup.find_all('form')) > 0
        has_javascript = 'script' in html_content.lower()
        has_images = len(soup.find_all('img')) > 0

        # Calculate navigation depth
        nav_depth = len(soup.find_all(['nav', 'ul', 'ol']))

        # Count content blocks
        content_blocks = len(soup.find_all(['div', 'section', 'article', 'main']))

        # Calculate text to HTML ratio
        text_length = len(soup.get_text())
        html_length = len(html_content)
        text_ratio = text_length / html_length if html_length > 0 else 0

        return PageStructure(
            has_tables=has_tables,
            has_forms=has_forms,
            has_javascript=has_javascript,
            has_images=has_images,
            navigation_depth=nav_depth,
            content_blocks=content_blocks,
            text_to_html_ratio=text_ratio
        )

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate semantic embedding using Gemini"""
        try:
            if not text.strip():
                return []

            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return []

    async def _classify_content(self, text: str, url: str) -> Tuple[str, float]:
        """Classify content type using pattern matching and LLM"""
        # Check URL patterns first
        url_lower = url.lower()
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    return content_type, 0.8

        # Use LLM for classification if no clear pattern match
        try:
            prompt = f"""
            Analyze this web page content and classify it into one of these categories:
            - article: News, blog posts, editorial content
            - product: E-commerce items, product listings
            - review: User reviews, ratings, feedback
            - directory: Listings, categories, navigation pages
            - profile: User profiles, personal pages
            - documentation: Guides, tutorials, help pages
            - other: Any other type of content

            Content: {text[:2000]}

            Respond with only the category name and confidence score (0-1) separated by a comma.
            Example: article,0.95
            """

            response = self.model.generate_content(prompt)
            result = response.text.strip().split(',')

            if len(result) == 2:
                category = result[0].strip().lower()
                confidence = float(result[1].strip())
                return category, min(confidence, 1.0)

        except Exception as e:
            logger.error(f"Error in content classification: {str(e)}")

        return "unknown", 0.5

    def _generate_fingerprint(self, url: str, text: str) -> str:
        """Generate a unique fingerprint for the page"""
        content_hash = hashlib.md5(f"{url}{text}".encode()).hexdigest()
        return content_hash

    def _determine_strategy(self, structure: PageStructure, content_type: str, confidence: float) -> str:
        """Determine the best scraping strategy based on analysis"""
        # High confidence in content type suggests structured data
        if confidence > 0.8:
            return "scrapegraph"

        # Dynamic content with JavaScript suggests Crawl4AI
        if structure.has_javascript and structure.dynamic_content_ratio > 0.3:
            return "crawl4ai"

        # Tables suggest structured data extraction
        if structure.has_tables:
            return "scrapegraph"

        # Complex navigation suggests crawling
        if structure.navigation_depth > 5:
            return "scrapy"

        # Default to ScrapeGraphAI for semantic understanding
        return "scrapegraph"

    def _calculate_structure_score(self, structure: PageStructure) -> float:
        """Calculate a score representing page structure quality"""
        score = 0.0

        if structure.has_tables:
            score += 0.3
        if structure.has_forms:
            score += 0.2
        if structure.has_images:
            score += 0.1
        if structure.text_to_html_ratio > 0.1:
            score += 0.4

        return min(score, 1.0)

    def _calculate_dynamic_ratio(self, html_content: str) -> float:
        """Estimate the ratio of dynamic content"""
        # Count JavaScript-related elements
        js_indicators = [
            '<script', 'javascript:', 'onclick', 'onload',
            'vue', 'react', 'angular', 'jquery'
        ]

        dynamic_count = 0
        for indicator in js_indicators:
            dynamic_count += html_content.lower().count(indicator)

        # Estimate dynamic content ratio
        total_length = len(html_content)
        if total_length == 0:
            return 0.0

        return min(dynamic_count * 10 / total_length, 1.0)

    def _extract_features(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract additional features for analysis"""
        features = {}

        # Extract meta information
        meta_tags = soup.find_all('meta')
        features['meta_count'] = len(meta_tags)

        # Check for common frameworks
        html_content = str(soup).lower()
        features['has_react'] = 'react' in html_content
        features['has_vue'] = 'vue' in html_content
        features['has_angular'] = 'angular' in html_content

        # Extract domain information
        parsed_url = urlparse(url)
        features['domain'] = parsed_url.netloc
        features['path_depth'] = len(parsed_url.path.split('/')) - 1

        # Count various elements
        features['link_count'] = len(soup.find_all('a'))
        features['image_count'] = len(soup.find_all('img'))
        features['heading_count'] = len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))

        return features

    # Learning System Methods
    def _load_semantic_memory(self) -> Dict[str, Any]:
        """Load semantic memory from persistent storage"""
        memory_file = self.knowledge_base_path / "semantic_memory.pkl"
        if memory_file.exists():
            try:
                with open(memory_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load semantic memory: {e}")
        return {
            'embeddings': [],
            'content_types': [],
            'success_patterns': defaultdict(list),
            'domain_patterns': defaultdict(dict)
        }

    def _load_pattern_library(self) -> Dict[str, List[str]]:
        """Load extraction pattern library"""
        pattern_file = self.knowledge_base_path / "pattern_library.json"
        if pattern_file.exists():
            try:
                with open(pattern_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load pattern library: {e}")
        return defaultdict(list)

    def _save_semantic_memory(self):
        """Save semantic memory to persistent storage"""
        try:
            memory_file = self.knowledge_base_path / "semantic_memory.pkl"
            with open(memory_file, 'wb') as f:
                pickle.dump(self.semantic_memory, f)
        except Exception as e:
            logger.error(f"Failed to save semantic memory: {e}")

    def _save_pattern_library(self):
        """Save pattern library to persistent storage"""
        try:
            pattern_file = self.knowledge_base_path / "pattern_library.json"
            with open(pattern_file, 'w') as f:
                json.dump(dict(self.pattern_library), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save pattern library: {e}")

    async def _classify_content_enhanced(self, text: str, url: str, soup: BeautifulSoup, embedding: List[float]) -> Tuple[str, float]:
        """Enhanced content classification using semantic understanding and learning"""
        # First try pattern-based classification
        url_lower = url.lower()
        best_match = None
        best_confidence = 0.0
        
        for content_type, patterns in self.content_patterns.items():
            # Check URL keywords
            keyword_matches = 0
            for pattern in patterns['keywords']:
                if re.search(pattern, url_lower):
                    keyword_matches += 1
            
            # Check semantic indicators in content
            semantic_matches = 0
            text_lower = text.lower()
            for indicator in patterns['semantic_indicators']:
                if indicator.lower() in text_lower:
                    semantic_matches += 1
            
            # Calculate confidence based on matches
            total_patterns = len(patterns['keywords']) + len(patterns['semantic_indicators'])
            confidence = (keyword_matches * 2 + semantic_matches) / (total_patterns * 1.5)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = content_type
        
        # Use semantic similarity with learned patterns
        if embedding and self.semantic_memory['embeddings']:
            semantic_confidence = await self._get_semantic_similarity_confidence(embedding)
            if semantic_confidence > best_confidence:
                best_confidence = semantic_confidence
                best_match = await self._get_most_similar_content_type(embedding)
        
        # Fallback to LLM classification with enhanced prompt
        if best_confidence < 0.6:
            try:
                # Extract key features for LLM analysis
                key_features = self._extract_classification_features(soup, url)
                
                prompt = f"""
                Analyze this web page and classify its content type. Consider both the URL structure and content features.
                
                URL: {url}
                Key Features: {json.dumps(key_features, indent=2)}
                Content Sample: {text[:1500]}
                
                Content Types:
                - article: News articles, blog posts, editorial content
                - product: E-commerce products, item listings
                - review: User reviews, ratings, testimonials
                - directory: Category pages, listings, navigation
                - profile: User profiles, personal pages, bios
                - documentation: Guides, tutorials, API docs, help pages
                - forum: Discussion threads, Q&A, community posts
                - media: Video pages, image galleries, multimedia content
                - other: Any other content type
                
                Respond with: content_type,confidence_score
                Example: article,0.92
                """
                
                response = self.model.generate_content(prompt)
                result = response.text.strip().split(',')
                
                if len(result) == 2:
                    llm_type = result[0].strip().lower()
                    llm_confidence = float(result[1].strip())
                    
                    if llm_confidence > best_confidence:
                        best_match = llm_type
                        best_confidence = llm_confidence
                        
            except Exception as e:
                logger.error(f"Error in LLM classification: {str(e)}")
        
        # Learn from this classification
        if best_match and best_confidence > 0.7:
            await self._update_learning_memory(embedding, best_match, url, best_confidence)
        
        return best_match or "unknown", min(best_confidence, 1.0)

    def _extract_classification_features(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract key features for content classification"""
        features = {}
        
        # URL analysis
        parsed_url = urlparse(url)
        features['domain'] = parsed_url.netloc
        features['path_segments'] = parsed_url.path.split('/')
        
        # Meta tags
        meta_description = soup.find('meta', attrs={'name': 'description'})
        features['meta_description'] = meta_description.get('content', '') if meta_description else ''
        
        # Title analysis
        title = soup.find('title')
        features['title'] = title.get_text() if title else ''
        
        # Structural elements
        features['has_article_tag'] = bool(soup.find('article'))
        features['has_product_schema'] = 'product' in str(soup).lower()
        features['has_review_schema'] = 'review' in str(soup).lower()
        features['has_breadcrumbs'] = bool(soup.find(attrs={'class': re.compile(r'breadcrumb', re.I)}))
        
        # Content indicators
        features['has_price'] = bool(re.search(r'\$[0-9]+|price', str(soup), re.I))
        features['has_rating'] = bool(re.search(r'rating|stars?|score', str(soup), re.I))
        features['has_author'] = bool(re.search(r'author|by\s+\w+', str(soup), re.I))
        features['has_date'] = bool(re.search(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}', str(soup)))
        
        return features

    async def _get_semantic_similarity_confidence(self, embedding: List[float]) -> float:
        """Calculate confidence based on semantic similarity to known patterns"""
        if not self.semantic_memory['embeddings'] or not embedding:
            return 0.0
        
        try:
            # Convert to numpy arrays for similarity calculation
            query_embedding = np.array(embedding).reshape(1, -1)
            stored_embeddings = np.array(self.semantic_memory['embeddings'])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, stored_embeddings)[0]
            
            # Return the maximum similarity as confidence
            return float(np.max(similarities))
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0

    async def _get_most_similar_content_type(self, embedding: List[float]) -> str:
        """Get the content type of the most similar stored embedding"""
        if not self.semantic_memory['embeddings'] or not embedding:
            return "unknown"
        
        try:
            query_embedding = np.array(embedding).reshape(1, -1)
            stored_embeddings = np.array(self.semantic_memory['embeddings'])
            
            similarities = cosine_similarity(query_embedding, stored_embeddings)[0]
            most_similar_idx = np.argmax(similarities)
            
            return self.semantic_memory['content_types'][most_similar_idx]
            
        except Exception as e:
            logger.error(f"Error finding most similar content type: {e}")
            return "unknown"

    async def _update_learning_memory(self, embedding: List[float], content_type: str, url: str, confidence: float):
        """Update the learning memory with new successful classification"""
        try:
            # Store embedding and content type
            self.semantic_memory['embeddings'].append(embedding)
            self.semantic_memory['content_types'].append(content_type)
            
            # Store success pattern
            domain = urlparse(url).netloc
            self.semantic_memory['success_patterns'][content_type].append({
                'domain': domain,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
            
            # Limit memory size to prevent unbounded growth
            max_memory_size = 1000
            if len(self.semantic_memory['embeddings']) > max_memory_size:
                # Remove oldest entries
                self.semantic_memory['embeddings'] = self.semantic_memory['embeddings'][-max_memory_size:]
                self.semantic_memory['content_types'] = self.semantic_memory['content_types'][-max_memory_size:]
            
            # Save to persistent storage periodically
            if len(self.semantic_memory['embeddings']) % 10 == 0:
                self._save_semantic_memory()
                
        except Exception as e:
            logger.error(f"Error updating learning memory: {e}")

    def _generate_semantic_fingerprint(self, url: str, text: str, embedding: List[float]) -> str:
        """Generate a semantic fingerprint combining content hash and embedding signature"""
        # Traditional content hash
        content_hash = hashlib.md5(f"{url}{text}".encode()).hexdigest()[:16]
        
        # Embedding signature (hash of first few embedding values)
        if embedding:
            embedding_signature = hashlib.md5(
                str(embedding[:10]).encode()
            ).hexdigest()[:16]
        else:
            embedding_signature = "0" * 16
        
        # Combine both for semantic fingerprint
        return f"{content_hash}_{embedding_signature}"

    def _determine_strategy_enhanced(self, structure: PageStructure, content_type: str, confidence: float, url: str) -> str:
        """Enhanced strategy determination with learning from past successes"""
        domain = urlparse(url).netloc
        
        # Check learned patterns for this domain
        if domain in self.success_metrics:
            domain_strategies = self.success_metrics[domain]
            if domain_strategies:
                # Use the most successful strategy for this domain
                best_strategy = max(domain_strategies, key=lambda x: x['success_rate'])
                if best_strategy['success_rate'] > 0.7:
                    return best_strategy['strategy']
        
        # Enhanced rule-based strategy selection
        if confidence > 0.9 and content_type in ['product', 'article']:
            return "scrapegraph"  # High confidence structured content
        
        if structure.has_javascript and structure.dynamic_content_ratio > 0.4:
            return "crawl4ai"  # Heavy JavaScript content
        
        if content_type == "directory" and structure.navigation_depth > 3:
            return "scrapy"  # Complex navigation structures
        
        if structure.has_tables and content_type in ['product', 'review']:
            return "scrapegraph"  # Structured data extraction
        
        # Default intelligent choice
        return "scrapegraph"

    def _extract_features_enhanced(self, soup: BeautifulSoup, url: str, embedding: List[float]) -> Dict[str, Any]:
        """Extract enhanced features including semantic analysis"""
        features = self._extract_features(soup, url)
        
        # Add semantic features
        if embedding:
            features['embedding_magnitude'] = float(np.linalg.norm(embedding)) if embedding else 0.0
            features['embedding_dimensions'] = len(embedding)
        
        # Enhanced content analysis
        features['content_complexity'] = self._calculate_content_complexity(soup)
        features['semantic_density'] = self._calculate_semantic_density(soup)
        features['information_architecture'] = self._analyze_information_architecture(soup)
        
        return features

    def _calculate_quality_score(self, structure: PageStructure, confidence: float, features: Dict[str, Any]) -> float:
        """Calculate overall quality score for the analysis"""
        score = 0.0
        
        # Confidence contributes 40%
        score += confidence * 0.4
        
        # Structure quality contributes 30%
        structure_quality = (
            (0.2 if structure.has_tables else 0) +
            (0.2 if structure.text_to_html_ratio > 0.1 else 0) +
            (0.3 if structure.content_blocks > 5 else structure.content_blocks * 0.06) +
            (0.3 if not structure.has_javascript else 0.1)  # Prefer static content for reliability
        )
        score += structure_quality * 0.3
        
        # Feature richness contributes 30%
        feature_score = min(len(features) / 20.0, 1.0)  # Normalize to 0-1
        score += feature_score * 0.3
        
        return min(score, 1.0)

    def _extract_patterns(self, soup: BeautifulSoup, content_type: str) -> List[str]:
        """Extract extraction patterns for learning"""
        patterns = []
        
        # Extract common selectors based on content type
        if content_type == "article":
            patterns.extend(self._extract_article_patterns(soup))
        elif content_type == "product":
            patterns.extend(self._extract_product_patterns(soup))
        elif content_type == "review":
            patterns.extend(self._extract_review_patterns(soup))
        
        return patterns

    def _extract_article_patterns(self, soup: BeautifulSoup) -> List[str]:
        """Extract patterns specific to articles"""
        patterns = []
        
        # Title patterns
        for tag in ['h1', 'h2']:
            elements = soup.find_all(tag)
            for elem in elements[:3]:  # Limit to first 3
                if elem.get('class'):
                    patterns.append(f"{tag}.{' '.join(elem['class'])}")
        
        # Content patterns
        article_tag = soup.find('article')
        if article_tag and article_tag.get('class'):
            patterns.append(f"article.{' '.join(article_tag['class'])}")
        
        return patterns

    def _extract_product_patterns(self, soup: BeautifulSoup) -> List[str]:
        """Extract patterns specific to products"""
        patterns = []
        
        # Price patterns
        price_elements = soup.find_all(attrs={'class': re.compile(r'price', re.I)})
        for elem in price_elements[:2]:
            if elem.get('class'):
                patterns.append(f"{elem.name}.{' '.join(elem['class'])}")
        
        return patterns

    def _extract_review_patterns(self, soup: BeautifulSoup) -> List[str]:
        """Extract patterns specific to reviews"""
        patterns = []
        
        # Rating patterns
        rating_elements = soup.find_all(attrs={'class': re.compile(r'rating|star', re.I)})
        for elem in rating_elements[:2]:
            if elem.get('class'):
                patterns.append(f"{elem.name}.{' '.join(elem['class'])}")
        
        return patterns

    async def _find_semantic_clusters(self, embedding: List[float], content_type: str) -> List[str]:
        """Find semantic clusters this content belongs to"""
        clusters = [content_type]  # Always include the primary content type
        
        if not embedding or not self.semantic_memory['embeddings']:
            return clusters
        
        try:
            # Find similar content types based on embedding similarity
            query_embedding = np.array(embedding).reshape(1, -1)
            stored_embeddings = np.array(self.semantic_memory['embeddings'])
            
            similarities = cosine_similarity(query_embedding, stored_embeddings)[0]
            
            # Find highly similar content (similarity > 0.8)
            similar_indices = np.where(similarities > 0.8)[0]
            
            for idx in similar_indices:
                similar_type = self.semantic_memory['content_types'][idx]
                if similar_type not in clusters:
                    clusters.append(similar_type)
            
        except Exception as e:
            logger.error(f"Error finding semantic clusters: {e}")
        
        return clusters[:5]  # Limit to 5 clusters

    def _calculate_content_freshness(self, soup: BeautifulSoup, features: Dict[str, Any]) -> float:
        """Calculate how fresh/recent the content appears to be"""
        freshness = 0.5  # Default neutral freshness
        
        # Look for date indicators
        current_year = datetime.now().year
        
        # Check for recent dates in content
        date_patterns = [
            r'\b' + str(current_year) + r'\b',
            r'\b' + str(current_year - 1) + r'\b',
            r'\bupdated\b.*\b' + str(current_year) + r'\b'
        ]
        
        content_text = soup.get_text().lower()
        for pattern in date_patterns:
            if re.search(pattern, content_text, re.I):
                freshness += 0.2
        
        # Check meta tags for freshness indicators
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            content = meta.get('content', '').lower()
            if any(word in content for word in ['updated', 'modified', str(current_year)]):
                freshness += 0.1
        
        return min(freshness, 1.0)

    def _calculate_content_complexity(self, soup: BeautifulSoup) -> float:
        """Calculate content complexity score"""
        # Count different types of elements
        element_counts = {
            'divs': len(soup.find_all('div')),
            'spans': len(soup.find_all('span')),
            'tables': len(soup.find_all('table')),
            'forms': len(soup.find_all('form')),
            'lists': len(soup.find_all(['ul', 'ol'])),
            'media': len(soup.find_all(['img', 'video', 'audio']))
        }
        
        # Calculate complexity based on element diversity and count
        total_elements = sum(element_counts.values())
        element_types = len([count for count in element_counts.values() if count > 0])
        
        if total_elements == 0:
            return 0.0
        
        complexity = (element_types / 6.0) * 0.5 + min(total_elements / 100.0, 1.0) * 0.5
        return min(complexity, 1.0)

    def _calculate_semantic_density(self, soup: BeautifulSoup) -> float:
        """Calculate semantic density of the content"""
        # Count semantic HTML5 elements
        semantic_elements = soup.find_all([
            'article', 'section', 'nav', 'aside', 'header', 'footer',
            'main', 'figure', 'figcaption', 'time', 'mark'
        ])
        
        total_elements = len(soup.find_all())
        if total_elements == 0:
            return 0.0
        
        return min(len(semantic_elements) / total_elements, 1.0)

    def _analyze_information_architecture(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze the information architecture of the page"""
        architecture = {
            'heading_hierarchy': self._analyze_heading_hierarchy(soup),
            'navigation_structure': self._analyze_navigation_structure(soup),
            'content_organization': self._analyze_content_organization(soup)
        }
        return architecture

    def _analyze_heading_hierarchy(self, soup: BeautifulSoup) -> Dict[str, int]:
        """Analyze heading hierarchy"""
        headings = {}
        for i in range(1, 7):
            headings[f'h{i}'] = len(soup.find_all(f'h{i}'))
        return headings

    def _analyze_navigation_structure(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze navigation structure"""
        nav_elements = soup.find_all('nav')
        breadcrumbs = soup.find_all(attrs={'class': re.compile(r'breadcrumb', re.I)})
        
        return {
            'nav_count': len(nav_elements),
            'has_breadcrumbs': len(breadcrumbs) > 0,
            'menu_depth': self._calculate_menu_depth(nav_elements)
        }

    def _calculate_menu_depth(self, nav_elements: List) -> int:
        """Calculate maximum menu depth"""
        max_depth = 0
        for nav in nav_elements:
            depth = self._get_nested_list_depth(nav)
            max_depth = max(max_depth, depth)
        return max_depth

    def _get_nested_list_depth(self, element) -> int:
        """Get the depth of nested lists"""
        lists = element.find_all(['ul', 'ol'])
        if not lists:
            return 0
        
        max_depth = 1
        for lst in lists:
            nested_lists = lst.find_all(['ul', 'ol'])
            if nested_lists:
                max_depth = max(max_depth, 1 + self._get_nested_list_depth(lst))
        
        return max_depth

    def _analyze_content_organization(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze content organization"""
        return {
            'sections': len(soup.find_all('section')),
            'articles': len(soup.find_all('article')),
            'asides': len(soup.find_all('aside')),
            'has_main': bool(soup.find('main')),
            'content_blocks': len(soup.find_all(['div', 'section', 'article']))
        }

    def record_scraping_success(self, url: str, strategy: str, success: bool, quality_score: float = 0.0):
        """Record the success/failure of a scraping attempt for learning"""
        domain = urlparse(url).netloc
        
        if domain not in self.success_metrics:
            self.success_metrics[domain] = []
        
        # Find existing strategy record or create new one
        strategy_record = None
        for record in self.success_metrics[domain]:
            if record['strategy'] == strategy:
                strategy_record = record
                break
        
        if not strategy_record:
            strategy_record = {
                'strategy': strategy,
                'attempts': 0,
                'successes': 0,
                'success_rate': 0.0,
                'avg_quality': 0.0,
                'quality_scores': []
            }
            self.success_metrics[domain].append(strategy_record)
        
        # Update metrics
        strategy_record['attempts'] += 1
        if success:
            strategy_record['successes'] += 1
            if quality_score > 0:
                strategy_record['quality_scores'].append(quality_score)
        
        # Recalculate success rate and average quality
        strategy_record['success_rate'] = strategy_record['successes'] / strategy_record['attempts']
        if strategy_record['quality_scores']:
            strategy_record['avg_quality'] = sum(strategy_record['quality_scores']) / len(strategy_record['quality_scores'])
        
        # Save metrics periodically
        if strategy_record['attempts'] % 5 == 0:
            self._save_success_metrics()

    def _save_success_metrics(self):
        """Save success metrics to persistent storage"""
        try:
            metrics_file = self.knowledge_base_path / "success_metrics.json"
            # Convert defaultdict to regular dict for JSON serialization
            metrics_dict = {k: v for k, v in self.success_metrics.items()}
            with open(metrics_file, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save success metrics: {e}")

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning system"""
        insights = {
            'total_learned_patterns': len(self.semantic_memory['embeddings']),
            'content_type_distribution': {},
            'domain_strategies': {},
            'top_performing_strategies': []
        }
        
        # Content type distribution
        for content_type in self.semantic_memory['content_types']:
            insights['content_type_distribution'][content_type] = insights['content_type_distribution'].get(content_type, 0) + 1
        
        # Domain strategies
        for domain, strategies in self.success_metrics.items():
            best_strategy = max(strategies, key=lambda x: x['success_rate']) if strategies else None
            if best_strategy:
                insights['domain_strategies'][domain] = {
                    'best_strategy': best_strategy['strategy'],
                    'success_rate': best_strategy['success_rate'],
                    'avg_quality': best_strategy['avg_quality']
                }
        
        # Top performing strategies overall
        all_strategies = []
        for domain_strategies in self.success_metrics.values():
            all_strategies.extend(domain_strategies)
        
        if all_strategies:
            top_strategies = sorted(all_strategies, key=lambda x: x['success_rate'], reverse=True)[:5]
            insights['top_performing_strategies'] = [
                {
                    'strategy': s['strategy'],
                    'success_rate': s['success_rate'],
                    'attempts': s['attempts'],
                    'avg_quality': s['avg_quality']
                }
                for s in top_strategies
            ]
        
        return insights