from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import asyncio
import json
import psutil
import time
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from models import ScrapingJob, ScrapingResult, QualityMetric, AntiDetectionLog
from database import get_db_connection
import sqlite3

logger = logging.getLogger(__name__)
router = APIRouter()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.monitoring_data = {
            'active_jobs': {},
            'system_health': {},
            'performance_data': [],
            'quality_metrics': {}
        }
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
        # Send initial data
        await self.send_initial_data(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_initial_data(self, websocket: WebSocket):
        """Send current monitoring data to newly connected client"""
        try:
            # Send system health
            health_data = await self.get_system_health()
            await websocket.send_text(json.dumps({
                'type': 'system_health',
                'health': health_data
            }))
            
            # Send active jobs
            active_jobs = await self.get_active_jobs()
            for job in active_jobs:
                await websocket.send_text(json.dumps({
                    'type': 'job_update',
                    'job': job
                }))
            
            # Send quality metrics
            quality_metrics = await self.get_quality_metrics()
            await websocket.send_text(json.dumps({
                'type': 'quality_metrics',
                'metrics': quality_metrics
            }))
            
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
            
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get job statistics from database
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Active jobs count
            cursor.execute("""
                SELECT COUNT(*) FROM scraping_jobs 
                WHERE status IN ('pending', 'running')
            """)
            active_jobs = cursor.fetchone()[0]
            
            # Queued jobs count
            cursor.execute("""
                SELECT COUNT(*) FROM scraping_jobs 
                WHERE status = 'pending'
            """)
            queued_jobs = cursor.fetchone()[0]
            
            # Today's jobs count
            today = datetime.now().date()
            cursor.execute("""
                SELECT COUNT(*) FROM scraping_jobs 
                WHERE DATE(created_at) = ?
            """, (today,))
            total_jobs_today = cursor.fetchone()[0]
            
            # Success rate calculation
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                    COUNT(*) as total
                FROM scraping_jobs 
                WHERE DATE(created_at) = ?
            """, (today,))
            result = cursor.fetchone()
            success_rate = (result[0] / result[1] * 100) if result[1] > 0 else 0
            
            # Average response time
            cursor.execute("""
                SELECT AVG(processing_time) FROM scraping_jobs 
                WHERE status = 'completed' AND DATE(created_at) = ?
            """, (today,))
            avg_response_time = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'cpu': round(cpu_percent, 1),
                'memory': round(memory.percent, 1),
                'disk': round(disk.percent, 1),
                'network': 0,  # Placeholder for network usage
                'activeJobs': active_jobs,
                'queuedJobs': queued_jobs,
                'totalJobsToday': total_jobs_today,
                'successRate': round(success_rate, 1),
                'avgResponseTime': round(avg_response_time, 2)
            }
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                'cpu': 0, 'memory': 0, 'disk': 0, 'network': 0,
                'activeJobs': 0, 'queuedJobs': 0, 'totalJobsToday': 0,
                'successRate': 0, 'avgResponseTime': 0
            }
    
    async def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get currently active scraping jobs"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    sj.id, sj.url, sj.status, sj.strategy, sj.created_at,
                    sj.quality_score, sj.processing_time,
                    COUNT(sr.id) as data_extracted,
                    CASE WHEN adl.id IS NOT NULL THEN 1 ELSE 0 END as anti_detection_triggered
                FROM scraping_jobs sj
                LEFT JOIN scraping_results sr ON sj.id = sr.job_id
                LEFT JOIN anti_detection_logs adl ON sj.id = adl.job_id
                WHERE sj.status IN ('pending', 'running')
                GROUP BY sj.id
                ORDER BY sj.created_at DESC
            """)
            
            jobs = []
            for row in cursor.fetchall():
                # Calculate progress based on status and processing time
                progress = 0
                if row[2] == 'running':
                    # Estimate progress based on processing time (rough estimate)
                    processing_time = row[6] or 0
                    progress = min(90, processing_time * 10)  # Cap at 90% until completion
                elif row[2] == 'pending':
                    progress = 0
                
                jobs.append({
                    'id': str(row[0]),
                    'url': row[1],
                    'status': row[2],
                    'strategy': row[3] or 'auto',
                    'startTime': row[4],
                    'progress': progress,
                    'qualityScore': row[5] or 0,
                    'dataExtracted': row[7],
                    'antiDetectionTriggered': bool(row[8])
                })
            
            conn.close()
            return jobs
        except Exception as e:
            logger.error(f"Error getting active jobs: {e}")
            return []
    
    async def get_quality_metrics(self) -> Dict[str, float]:
        """Get current data quality metrics"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get average quality metrics from recent jobs
            cursor.execute("""
                SELECT 
                    AVG(qm.completeness) as completeness,
                    AVG(qm.accuracy) as accuracy,
                    AVG(qm.relevance) as relevance,
                    AVG(qm.structure_score) as structure,
                    AVG(qm.freshness) as freshness
                FROM quality_metrics qm
                JOIN scraping_jobs sj ON qm.job_id = sj.id
                WHERE sj.created_at >= datetime('now', '-24 hours')
            """)
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] is not None:
                return {
                    'completeness': round(result[0], 1),
                    'accuracy': round(result[1], 1),
                    'relevance': round(result[2], 1),
                    'structure': round(result[3], 1),
                    'freshness': round(result[4], 1)
                }
            else:
                return {
                    'completeness': 0, 'accuracy': 0, 'relevance': 0,
                    'structure': 0, 'freshness': 0
                }
        except Exception as e:
            logger.error(f"Error getting quality metrics: {e}")
            return {
                'completeness': 0, 'accuracy': 0, 'relevance': 0,
                'structure': 0, 'freshness': 0
            }
    
    async def update_job_status(self, job_id: str, status: str, progress: int = None, quality_score: float = None):
        """Update job status and broadcast to clients"""
        try:
            # Get updated job data
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    sj.id, sj.url, sj.status, sj.strategy, sj.created_at,
                    sj.quality_score, sj.processing_time,
                    COUNT(sr.id) as data_extracted,
                    CASE WHEN adl.id IS NOT NULL THEN 1 ELSE 0 END as anti_detection_triggered
                FROM scraping_jobs sj
                LEFT JOIN scraping_results sr ON sj.id = sr.job_id
                LEFT JOIN anti_detection_logs adl ON sj.id = adl.job_id
                WHERE sj.id = ?
                GROUP BY sj.id
            """, (job_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                job_data = {
                    'id': str(row[0]),
                    'url': row[1],
                    'status': row[2],
                    'strategy': row[3] or 'auto',
                    'startTime': row[4],
                    'progress': progress or (100 if status == 'completed' else 0),
                    'qualityScore': quality_score or row[5] or 0,
                    'dataExtracted': row[7],
                    'antiDetectionTriggered': bool(row[8])
                }
                
                await self.broadcast({
                    'type': 'job_update',
                    'job': job_data
                })
        except Exception as e:
            logger.error(f"Error updating job status: {e}")

# Global connection manager instance
manager = ConnectionManager()

@router.websocket("/ws/monitoring")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Handle any client messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@router.get("/api/monitoring/status")
async def get_monitoring_status():
    """Get current monitoring status (fallback for when WebSocket is not available)"""
    try:
        system_health = await manager.get_system_health()
        active_jobs = await manager.get_active_jobs()
        quality_metrics = await manager.get_quality_metrics()
        
        # Generate sample performance data point
        performance = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'successRate': system_health['successRate'],
            'avgExecutionTime': system_health['avgResponseTime'],
            'dataQuality': sum(quality_metrics.values()) / len(quality_metrics) if quality_metrics else 0,
            'jobsCompleted': system_health['totalJobsToday']
        }
        
        return JSONResponse({
            'systemHealth': system_health,
            'activeJobs': active_jobs,
            'qualityMetrics': quality_metrics,
            'performance': performance
        })
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/monitoring/performance")
async def get_performance_data(hours: int = 24):
    """Get performance data for the specified time period"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get hourly performance data
        cursor.execute("""
            SELECT 
                strftime('%H:00', created_at) as hour,
                COUNT(*) as total_jobs,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_jobs,
                AVG(processing_time) as avg_time,
                AVG(quality_score) as avg_quality
            FROM scraping_jobs 
            WHERE created_at >= datetime('now', '-{} hours')
            GROUP BY strftime('%H', created_at)
            ORDER BY hour
        """.format(hours))
        
        performance_data = []
        for row in cursor.fetchall():
            success_rate = (row[2] / row[1] * 100) if row[1] > 0 else 0
            performance_data.append({
                'timestamp': row[0],
                'successRate': round(success_rate, 1),
                'avgExecutionTime': round(row[3] or 0, 2),
                'dataQuality': round(row[4] or 0, 1),
                'jobsCompleted': row[2]
            })
        
        conn.close()
        return JSONResponse(performance_data)
    except Exception as e:
        logger.error(f"Error getting performance data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/monitoring/analytics")
async def get_analytics_data():
    """Get analytics data for dashboard"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Strategy distribution
        cursor.execute("""
            SELECT strategy, COUNT(*) as count
            FROM scraping_jobs 
            WHERE created_at >= datetime('now', '-7 days')
            GROUP BY strategy
        """)
        strategy_distribution = [{'name': row[0] or 'auto', 'value': row[1]} for row in cursor.fetchall()]
        
        # Content type distribution
        cursor.execute("""
            SELECT 
                cc.content_type, 
                COUNT(*) as count
            FROM content_classifications cc
            JOIN scraping_jobs sj ON cc.job_id = sj.id
            WHERE sj.created_at >= datetime('now', '-7 days')
            GROUP BY cc.content_type
        """)
        content_type_distribution = [{'name': row[0], 'value': row[1]} for row in cursor.fetchall()]
        
        # Error analysis
        cursor.execute("""
            SELECT 
                adl.detection_type,
                COUNT(*) as count
            FROM anti_detection_logs adl
            JOIN scraping_jobs sj ON adl.job_id = sj.id
            WHERE sj.created_at >= datetime('now', '-7 days')
            GROUP BY adl.detection_type
        """)
        error_distribution = [{'name': row[0], 'value': row[1]} for row in cursor.fetchall()]
        
        conn.close()
        
        return JSONResponse({
            'strategyDistribution': strategy_distribution,
            'contentTypeDistribution': content_type_distribution,
            'errorDistribution': error_distribution
        })
    except Exception as e:
        logger.error(f"Error getting analytics data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task to periodically update monitoring data
async def monitoring_background_task():
    """Background task to periodically broadcast monitoring updates"""
    while True:
        try:
            # Update system health every 5 seconds
            health_data = await manager.get_system_health()
            await manager.broadcast({
                'type': 'system_health',
                'health': health_data
            })
            
            # Update performance data every 30 seconds
            if int(time.time()) % 30 == 0:
                performance = {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'successRate': health_data['successRate'],
                    'avgExecutionTime': health_data['avgResponseTime'],
                    'dataQuality': 85,  # Placeholder
                    'jobsCompleted': health_data['totalJobsToday']
                }
                await manager.broadcast({
                    'type': 'performance_update',
                    'performance': performance
                })
            
            # Update quality metrics every 60 seconds
            if int(time.time()) % 60 == 0:
                quality_metrics = await manager.get_quality_metrics()
                await manager.broadcast({
                    'type': 'quality_metrics',
                    'metrics': quality_metrics
                })
            
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error in monitoring background task: {e}")
            await asyncio.sleep(5)

# Function to be called when job status changes
async def notify_job_update(job_id: str, status: str, progress: int = None, quality_score: float = None):
    """Notify monitoring system of job updates"""
    await manager.update_job_status(job_id, status, progress, quality_score)

# Export the manager for use in other modules
__all__ = ['router', 'manager', 'notify_job_update', 'monitoring_background_task']