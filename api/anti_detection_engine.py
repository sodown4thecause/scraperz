"""Advanced Anti-Detection Engine
Implements sophisticated anti-detection measures including browser fingerprint randomization,
human-like behavior patterns, and advanced proxy rotation.
"""

import os
import json
import asyncio
import time
import random
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import numpy as np
from collections import defaultdict
import httpx
from urllib.parse import urlparse
import base64
from fake_useragent import UserAgent
import platform

logger = logging.getLogger(__name__)

@dataclass
class BrowserFingerprint:
    """Browser fingerprint configuration"""
    user_agent: str
    viewport_width: int
    viewport_height: int
    screen_width: int
    screen_height: int
    color_depth: int
    timezone: str
    language: str
    platform: str
    webgl_vendor: str
    webgl_renderer: str
    canvas_fingerprint: str
    audio_fingerprint: str
    fonts: List[str]
    plugins: List[str]
    webrtc_ip: str
    hardware_concurrency: int
    device_memory: int
    connection_type: str
    battery_level: Optional[float] = None
    charging: Optional[bool] = None

@dataclass
class HumanBehaviorPattern:
    """Human-like behavior pattern configuration"""
    mouse_movements: List[Tuple[int, int]]
    scroll_patterns: List[Dict[str, Any]]
    typing_speed: float  # characters per second
    typing_variance: float  # variance in typing speed
    pause_patterns: List[float]  # pause durations
    click_patterns: List[Dict[str, Any]]
    reading_time: float  # time to read content
    interaction_delays: Dict[str, float]

@dataclass
class ProxyConfiguration:
    """Proxy configuration with health tracking"""
    proxy_url: str
    proxy_type: str  # http, https, socks4, socks5
    username: Optional[str] = None
    password: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    success_rate: float = 1.0
    last_used: Optional[datetime] = None
    response_time: float = 0.0
    failure_count: int = 0
    is_active: bool = True
    detection_events: List[str] = field(default_factory=list)

@dataclass
class AntiDetectionSession:
    """Session state for anti-detection measures"""
    session_id: str
    domain: str
    fingerprint: BrowserFingerprint
    behavior_pattern: HumanBehaviorPattern
    proxy_config: Optional[ProxyConfiguration]
    created_at: datetime
    last_activity: datetime
    request_count: int = 0
    detection_score: float = 0.0
    session_duration: timedelta = timedelta()
    cookies: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)

class AdvancedAntiDetectionEngine:
    """Advanced anti-detection engine with sophisticated evasion techniques"""
    
    PROXY_HEALTH_CHECK_URL = "http://httpbin.org/ip"
    PROXY_HEALTH_CHECK_TIMEOUT = 10  # seconds

    async def check_proxy_health(self, proxy: ProxyConfiguration) -> bool:
        """Check if a proxy is alive and working."""
        try:
            async with httpx.AsyncClient(proxies=self._get_proxy_config(proxy), timeout=self.PROXY_HEALTH_CHECK_TIMEOUT) as client:
                response = await client.get(self.PROXY_HEALTH_CHECK_URL)
                if response.status_code == 200:
                    proxy.is_active = True
                    return True
                else:
                    proxy.is_active = False
                    return False
        except Exception as e:
            logger.warning(f"Proxy {proxy.proxy_url} health check failed: {e}")
            proxy.is_active = False
            return False
    """Advanced anti-detection engine with sophisticated evasion techniques"""
    
    def __init__(self):
        self.sessions: Dict[str, AntiDetectionSession] = {}
        self.proxy_pool: List[ProxyConfiguration] = []
        self.fingerprint_templates: List[Dict[str, Any]] = []
        self.behavior_templates: List[Dict[str, Any]] = []
        self.domain_profiles: Dict[str, Dict[str, Any]] = {}
        self.detection_patterns: Dict[str, List[str]] = {}
        self.user_agent_generator = UserAgent()
        
        # Load configurations
        self._load_proxy_configurations()
        self._load_fingerprint_templates()
        self._load_behavior_templates()
        self._load_detection_patterns()
    
    def _load_proxy_configurations(self):
        """Load proxy configurations from environment or config file"""
        proxy_config_path = os.getenv('PROXY_CONFIG_PATH', 'config/proxies.json')
        if os.path.exists(proxy_config_path):
            try:
                with open(proxy_config_path, 'r') as f:
                    proxy_data = json.load(f)
                    for proxy_info in proxy_data.get('proxies', []):
                        self.proxy_pool.append(ProxyConfiguration(**proxy_info))
            except Exception as e:
                logger.warning(f"Failed to load proxy configurations: {e}")
        
        # Add environment-based proxies
        env_proxies = os.getenv('PROXY_LIST', '').split(',')
        for proxy in env_proxies:
            if proxy.strip():
                self.proxy_pool.append(ProxyConfiguration(
                    proxy_url=proxy.strip(),
                    proxy_type='http'
                ))
    
    def _load_fingerprint_templates(self):
        """Load browser fingerprint templates"""
        self.fingerprint_templates = [
            {
                'name': 'chrome_windows',
                'viewport_range': [(1366, 768), (1920, 1080), (1440, 900)],
                'screen_range': [(1366, 768), (1920, 1080), (2560, 1440)],
                'color_depths': [24, 32],
                'timezones': ['America/New_York', 'America/Los_Angeles', 'Europe/London'],
                'languages': ['en-US', 'en-GB', 'en-CA'],
                'platforms': ['Win32', 'Win64'],
                'webgl_vendors': ['Google Inc.', 'NVIDIA Corporation'],
                'webgl_renderers': ['ANGLE (NVIDIA GeForce GTX 1060)', 'ANGLE (Intel HD Graphics)']
            },
            {
                'name': 'firefox_linux',
                'viewport_range': [(1280, 720), (1920, 1080), (1366, 768)],
                'screen_range': [(1280, 720), (1920, 1080), (2560, 1440)],
                'color_depths': [24],
                'timezones': ['UTC', 'Europe/Berlin', 'America/New_York'],
                'languages': ['en-US', 'de-DE', 'fr-FR'],
                'platforms': ['Linux x86_64'],
                'webgl_vendors': ['Mozilla', 'Mesa'],
                'webgl_renderers': ['Mesa DRI Intel(R) HD Graphics', 'llvmpipe']
            }
        ]
    
    def _load_behavior_templates(self):
        """Load human behavior templates"""
        self.behavior_templates = [
            {
                'name': 'casual_reader',
                'typing_speed_range': (2.5, 4.0),
                'reading_speed': 250,  # words per minute
                'scroll_frequency': 'medium',
                'pause_probability': 0.3,
                'interaction_style': 'deliberate'
            },
            {
                'name': 'power_user',
                'typing_speed_range': (4.0, 6.5),
                'reading_speed': 400,
                'scroll_frequency': 'high',
                'pause_probability': 0.1,
                'interaction_style': 'efficient'
            },
            {
                'name': 'mobile_user',
                'typing_speed_range': (1.5, 3.0),
                'reading_speed': 200,
                'scroll_frequency': 'high',
                'pause_probability': 0.4,
                'interaction_style': 'touch_based'
            }
        ]
    
    def _load_detection_patterns(self):
        """Load known detection patterns"""
        self.detection_patterns = {
            'cloudflare': [
                'cf-ray',
                'cloudflare',
                'checking your browser',
                'ddos protection'
            ],
            'captcha': [
                'recaptcha',
                'hcaptcha',
                'solve the captcha',
                'verify you are human'
            ],
            'rate_limiting': [
                'too many requests',
                'rate limit exceeded',
                'slow down',
                '429'
            ],
            'bot_detection': [
                'automated traffic',
                'bot detected',
                'suspicious activity',
                'access denied'
            ]
        }
    
    async def create_session(self, domain: str, strategy_hint: Optional[str] = None) -> str:
        """Create a new anti-detection session"""
        session_id = str(uuid.uuid4())
        
        # Generate fingerprint
        fingerprint = self._generate_browser_fingerprint(domain, strategy_hint)
        
        # Generate behavior pattern
        behavior_pattern = self._generate_behavior_pattern(domain, strategy_hint)
        
        # Select proxy
        proxy_config = await self._select_optimal_proxy(domain)
        
        # Create session
        session = AntiDetectionSession(
            session_id=session_id,
            domain=domain,
            fingerprint=fingerprint,
            behavior_pattern=behavior_pattern,
            proxy_config=proxy_config,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created anti-detection session {session_id} for domain {domain}")
        
        return session_id
    
    def _generate_browser_fingerprint(self, domain: str, strategy_hint: Optional[str] = None) -> BrowserFingerprint:
        """Generate a realistic browser fingerprint"""
        # Select template based on domain profile or random
        template = random.choice(self.fingerprint_templates)
        
        # Generate user agent
        user_agent = self.user_agent_generator.random
        
        # Generate viewport and screen dimensions
        viewport = random.choice(template['viewport_range'])
        screen = random.choice(template['screen_range'])
        
        # Generate other fingerprint components
        fingerprint = BrowserFingerprint(
            user_agent=user_agent,
            viewport_width=viewport[0],
            viewport_height=viewport[1],
            screen_width=screen[0],
            screen_height=screen[1],
            color_depth=random.choice(template['color_depths']),
            timezone=random.choice(template['timezones']),
            language=random.choice(template['languages']),
            platform=random.choice(template['platforms']),
            webgl_vendor=random.choice(template['webgl_vendors']),
            webgl_renderer=random.choice(template['webgl_renderers']),
            canvas_fingerprint=self._generate_canvas_fingerprint(),
            audio_fingerprint=self._generate_audio_fingerprint(),
            fonts=self._generate_font_list(),
            plugins=self._generate_plugin_list(),
            webrtc_ip=self._generate_webrtc_ip(),
            hardware_concurrency=random.choice([2, 4, 8, 12, 16]),
            device_memory=random.choice([2, 4, 8, 16, 32]),
            connection_type=random.choice(['4g', 'wifi', 'ethernet']),
            battery_level=random.uniform(0.2, 1.0) if random.random() < 0.7 else None,
            charging=random.choice([True, False]) if random.random() < 0.7 else None
        )
        
        return fingerprint
    
    def _generate_behavior_pattern(self, domain: str, strategy_hint: Optional[str] = None) -> HumanBehaviorPattern:
        """Generate human-like behavior pattern"""
        template = random.choice(self.behavior_templates)
        
        # Generate typing characteristics
        typing_speed = random.uniform(*template['typing_speed_range'])
        typing_variance = typing_speed * 0.2
        
        # Generate mouse movements
        mouse_movements = self._generate_mouse_movements()
        
        # Generate scroll patterns
        scroll_patterns = self._generate_scroll_patterns(template['scroll_frequency'])
        
        # Generate pause patterns
        pause_patterns = self._generate_pause_patterns(template['pause_probability'])
        
        # Generate click patterns
        click_patterns = self._generate_click_patterns(template['interaction_style'])
        
        # Calculate reading time
        reading_time = 60 / template['reading_speed']  # seconds per word
        
        # Generate interaction delays
        interaction_delays = {
            'page_load': random.uniform(2.0, 5.0),
            'form_fill': random.uniform(0.5, 2.0),
            'link_click': random.uniform(0.3, 1.0),
            'scroll': random.uniform(0.1, 0.5)
        }
        
        return HumanBehaviorPattern(
            mouse_movements=mouse_movements,
            scroll_patterns=scroll_patterns,
            typing_speed=typing_speed,
            typing_variance=typing_variance,
            pause_patterns=pause_patterns,
            click_patterns=click_patterns,
            reading_time=reading_time,
            interaction_delays=interaction_delays
        )
    
    def _select_optimal_proxy(self, domain: str) -> Optional[ProxyConfiguration]:
        """Select the best proxy for the domain"""
        if not self.proxy_pool:
            return None
        
        # Filter active proxies
        active_proxies = [p for p in self.proxy_pool if p.is_active]
        if not active_proxies:
            return None
        
        # Score proxies based on success rate, response time, and recent usage
        scored_proxies = []
        for proxy in active_proxies:
            score = proxy.success_rate
            
            # Penalize slow proxies
            if proxy.response_time > 0:
                score *= max(0.1, 1.0 - (proxy.response_time / 10.0))
            
            # Prefer less recently used proxies
            if proxy.last_used:
                hours_since_use = (datetime.now() - proxy.last_used).total_seconds() / 3600
                score *= min(2.0, 1.0 + (hours_since_use / 24.0))
            
            # Penalize proxies with recent detection events
            recent_detections = len([e for e in proxy.detection_events 
                                   if (datetime.now() - datetime.fromisoformat(e.split('|')[0])).days < 1])
            score *= max(0.1, 1.0 - (recent_detections * 0.3))
            
            scored_proxies.append((score, proxy))
        
        # Select proxy with weighted random choice
        scored_proxies.sort(key=lambda x: x[0], reverse=True)
        weights = [score for score, _ in scored_proxies]
        
        if sum(weights) > 0:
            selected_proxy = np.random.choice(
                [proxy for _, proxy in scored_proxies],
                p=np.array(weights) / sum(weights)
            )
            return selected_proxy
        
        return random.choice(active_proxies)
    
    async def get_session_config(self, session_id: str) -> Dict[str, Any]:
        """Get configuration for a session"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        # Apply human-like delay
        await self._apply_human_delay(session)
        
        # Update session activity
        session.last_activity = datetime.now()
        session.request_count += 1
        
        # Generate headers
        headers = self._generate_realistic_headers(session.fingerprint)
        session.headers.update(headers)
        
        config = {
            'user_agent': session.fingerprint.user_agent,
            'headers': session.headers,
            'proxy': self._get_proxy_config(session.proxy_config),
            'viewport': {
                'width': session.fingerprint.viewport_width,
                'height': session.fingerprint.viewport_height
            },
            'fingerprint': {
                'canvas': session.fingerprint.canvas_fingerprint,
                'webgl_vendor': session.fingerprint.webgl_vendor,
                'webgl_renderer': session.fingerprint.webgl_renderer,
                'timezone': session.fingerprint.timezone,
                'language': session.fingerprint.language,
                'platform': session.fingerprint.platform,
                'hardware_concurrency': session.fingerprint.hardware_concurrency,
                'device_memory': session.fingerprint.device_memory
            },
            'behavior': {
                'typing_speed': session.behavior_pattern.typing_speed,
                'reading_time': session.behavior_pattern.reading_time,
                'interaction_delays': session.behavior_pattern.interaction_delays
            },
            'cookies': session.cookies
        }
        
        return config
    
    async def _apply_human_delay(self, session: AntiDetectionSession):
        """Apply human-like delays between requests"""
        if session.request_count > 0:
            # Base delay from behavior pattern
            base_delay = session.behavior_pattern.interaction_delays.get('page_load', 2.0)
            
            # Add randomness
            delay = random.uniform(base_delay * 0.5, base_delay * 1.5)
            
            # Increase delay if many recent requests
            if session.request_count > 10:
                delay *= 1.5
            
            # Add pause patterns
            if random.random() < 0.3:  # 30% chance of longer pause
                pause = random.choice(session.behavior_pattern.pause_patterns)
                delay += pause
            
            await asyncio.sleep(delay)
    
    def _generate_realistic_headers(self, fingerprint: BrowserFingerprint) -> Dict[str, str]:
        """Generate realistic HTTP headers"""
        headers = {
            'User-Agent': fingerprint.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': f"{fingerprint.language},en;q=0.5",
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        # Add platform-specific headers
        if 'Windows' in fingerprint.platform:
            headers['Sec-CH-UA-Platform'] = '"Windows"'
        elif 'Linux' in fingerprint.platform:
            headers['Sec-CH-UA-Platform'] = '"Linux"'
        elif 'Mac' in fingerprint.platform:
            headers['Sec-CH-UA-Platform'] = '"macOS"'
        
        # Add viewport hints
        headers['Sec-CH-Viewport-Width'] = str(fingerprint.viewport_width)
        headers['Sec-CH-Viewport-Height'] = str(fingerprint.viewport_height)
        
        return headers
    
    def _get_proxy_config(self, proxy: Optional[ProxyConfiguration]) -> Optional[Dict[str, Any]]:
        """Get proxy configuration for requests"""
        if not proxy:
            return None
        
        config = {
            'url': proxy.proxy_url,
            'type': proxy.proxy_type
        }
        
        if proxy.username and proxy.password:
            config['auth'] = {
                'username': proxy.username,
                'password': proxy.password
            }
        
        return config
    
    def _generate_canvas_fingerprint(self) -> str:
        """Generate a unique canvas fingerprint"""
        # Simulate canvas rendering variations
        base_data = f"canvas_{random.randint(1000, 9999)}_{time.time()}"
        return hashlib.md5(base_data.encode()).hexdigest()[:16]
    
    def _generate_audio_fingerprint(self) -> str:
        """Generate a unique audio fingerprint"""
        base_data = f"audio_{random.randint(1000, 9999)}_{time.time()}"
        return hashlib.md5(base_data.encode()).hexdigest()[:16]
    
    def _generate_font_list(self) -> List[str]:
        """Generate a realistic font list"""
        common_fonts = [
            'Arial', 'Helvetica', 'Times New Roman', 'Courier New',
            'Verdana', 'Georgia', 'Palatino', 'Garamond',
            'Bookman', 'Comic Sans MS', 'Trebuchet MS', 'Arial Black'
        ]
        return random.sample(common_fonts, random.randint(8, 12))
    
    def _generate_plugin_list(self) -> List[str]:
        """Generate a realistic plugin list"""
        common_plugins = [
            'Chrome PDF Plugin', 'Chrome PDF Viewer', 'Native Client',
            'Widevine Content Decryption Module', 'Flash Player'
        ]
        return random.sample(common_plugins, random.randint(2, 5))
    
    def _generate_webrtc_ip(self) -> str:
        """Generate a realistic WebRTC IP"""
        return f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
    
    def _generate_mouse_movements(self) -> List[Tuple[int, int]]:
        """Generate realistic mouse movement patterns"""
        movements = []
        x, y = random.randint(100, 800), random.randint(100, 600)
        
        for _ in range(random.randint(5, 15)):
            # Add some randomness to movement
            dx = random.randint(-50, 50)
            dy = random.randint(-50, 50)
            x = max(0, min(1200, x + dx))
            y = max(0, min(800, y + dy))
            movements.append((x, y))
        
        return movements
    
    def _generate_scroll_patterns(self, frequency: str) -> List[Dict[str, Any]]:
        """Generate realistic scroll patterns"""
        patterns = []
        
        if frequency == 'high':
            scroll_count = random.randint(8, 15)
        elif frequency == 'medium':
            scroll_count = random.randint(4, 8)
        else:
            scroll_count = random.randint(1, 4)
        
        for _ in range(scroll_count):
            patterns.append({
                'direction': random.choice(['down', 'up']),
                'distance': random.randint(100, 500),
                'duration': random.uniform(0.5, 2.0)
            })
        
        return patterns
    
    def _generate_pause_patterns(self, probability: float) -> List[float]:
        """Generate realistic pause patterns"""
        pauses = []
        
        for _ in range(random.randint(3, 8)):
            if random.random() < probability:
                pauses.append(random.uniform(1.0, 5.0))
            else:
                pauses.append(random.uniform(0.1, 0.5))
        
        return pauses
    
    def _generate_click_patterns(self, style: str) -> List[Dict[str, Any]]:
        """Generate realistic click patterns"""
        patterns = []
        
        if style == 'efficient':
            click_count = random.randint(3, 6)
            delay_range = (0.2, 0.8)
        elif style == 'deliberate':
            click_count = random.randint(2, 4)
            delay_range = (1.0, 3.0)
        else:  # touch_based
            click_count = random.randint(4, 8)
            delay_range = (0.5, 1.5)
        
        for _ in range(click_count):
            patterns.append({
                'x': random.randint(50, 1000),
                'y': random.randint(50, 700),
                'delay': random.uniform(*delay_range),
                'duration': random.uniform(0.1, 0.3)
            })
        
        return patterns
    
    async def detect_anti_scraping_measures(self, response_text: str, status_code: int, headers: Dict[str, str]) -> Dict[str, Any]:
        """Detect anti-scraping measures in response"""
        detection_results = {
            'detected': False,
            'measures': [],
            'confidence': 0.0,
            'recommended_actions': []
        }
        
        # Check for known detection patterns
        for measure_type, patterns in self.detection_patterns.items():
            for pattern in patterns:
                if pattern.lower() in response_text.lower():
                    detection_results['detected'] = True
                    detection_results['measures'].append(measure_type)
                    detection_results['confidence'] += 0.2
        
        # Check status codes
        if status_code in [403, 429, 503]:
            detection_results['detected'] = True
            detection_results['measures'].append('http_blocking')
            detection_results['confidence'] += 0.3
        
        # Check headers
        suspicious_headers = ['cf-ray', 'x-rate-limit', 'x-blocked']
        for header in suspicious_headers:
            if header in headers:
                detection_results['detected'] = True
                detection_results['measures'].append('header_based_detection')
                detection_results['confidence'] += 0.1
        
        # Generate recommendations
        if detection_results['detected']:
            if 'cloudflare' in detection_results['measures']:
                detection_results['recommended_actions'].extend([
                    'rotate_fingerprint',
                    'change_proxy',
                    'increase_delays'
                ])
            
            if 'rate_limiting' in detection_results['measures']:
                detection_results['recommended_actions'].extend([
                    'increase_delays',
                    'change_proxy'
                ])
            
            if 'captcha' in detection_results['measures']:
                detection_results['recommended_actions'].extend([
                    'rotate_fingerprint',
                    'change_proxy',
                    'manual_intervention'
                ])
        
        detection_results['confidence'] = min(1.0, detection_results['confidence'])
        
        return detection_results
    
    async def update_proxy_performance(self, proxy_url: str, success: bool, response_time: float):
        """Update proxy performance metrics"""
        for proxy in self.proxy_pool:
            if proxy.proxy_url == proxy_url:
                proxy.last_used = datetime.now()
                proxy.response_time = response_time
                
                if success:
                    proxy.success_rate = proxy.success_rate * 0.9 + 0.1
                    proxy.failure_count = 0
                else:
                    proxy.success_rate = proxy.success_rate * 0.9
                    proxy.failure_count += 1
                    
                    # Deactivate proxy if too many failures
                    if proxy.failure_count >= 5:
                        proxy.is_active = False
                        logger.warning(f"Deactivated proxy {proxy_url} due to repeated failures")
                
                break
    
    async def rotate_session_fingerprint(self, session_id: str):
        """Rotate fingerprint for an existing session"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        session.fingerprint = self._generate_browser_fingerprint(session.domain)
        session.behavior_pattern = self._generate_behavior_pattern(session.domain)
        
        logger.info(f"Rotated fingerprint for session {session_id}")
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            # Sessions expire after 24 hours of inactivity
            if (current_time - session.last_activity).total_seconds() > 86400:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session {session_id}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions"""
        active_sessions = len(self.sessions)
        total_requests = sum(session.request_count for session in self.sessions.values())
        avg_detection_score = np.mean([session.detection_score for session in self.sessions.values()]) if self.sessions else 0
        
        proxy_stats = {
            'total_proxies': len(self.proxy_pool),
            'active_proxies': len([p for p in self.proxy_pool if p.is_active]),
            'avg_success_rate': np.mean([p.success_rate for p in self.proxy_pool]) if self.proxy_pool else 0
        }
        
        return {
            'active_sessions': active_sessions,
            'total_requests': total_requests,
            'avg_detection_score': avg_detection_score,
            'proxy_stats': proxy_stats
        }        }en(self.sessions)
        total_requests = sum(session.request_count for session in self.sessions.values())
        avg_detection_score = np.mean([session.detection_score for session in self.sessions.values()]) if self.sessions else 0
        
        proxy_stats = {
            'total_proxies': len(self.proxy_pool),
            'active_proxies': len([p for p in self.proxy_pool if p.is_active]),
            'avg_success_rate': np.mean([p.success_rate for p in self.proxy_pool]) if self.proxy_pool else 0
        }
        
        return {
            'active_sessions': active_sessions,
            'total_requests': total_requests,
            'avg_detection_score': avg_detection_score,
            'proxy_stats': proxy_stats
        }f.sessions.values())
        avg_detection_score = np.mean([session.detection_score for session in self.sessions.values()]) if self.sessions else 0
        
        proxy_stats = {
            'total_proxies': len(self.proxy_pool),
            'active_proxies': len([p for p in self.proxy_pool if p.is_active]),
            'avg_success_rate': np.mean([p.success_rate for p in self.proxy_pool]) if self.proxy_pool else 0
        }
        
        return {
            'active_sessions': active_sessions,
            'total_requests': total_requests,
            'avg_detection_score': avg_detection_score,
            'proxy_stats': proxy_stats
        }