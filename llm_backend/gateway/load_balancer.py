"""
Load balancer for LLM services
Handles service routing and health tracking
"""

import random
from typing import Dict, List, Optional
from loguru import logger


class LoadBalancer:
    """Simple load balancer for LLM services"""

    def __init__(self, services: Dict):
        """
        Initialize load balancer

        Args:
            services: Dictionary of service configurations
        """
        self.services = services
        self.health_status = {name: True for name in services.keys()}
        self.request_counts = {name: 0 for name in services.keys()}

    def get_service(self, model_name: str, strategy: str = "least_loaded") -> Optional[str]:
        """
        Get service URL based on load balancing strategy

        Args:
            model_name: Model name to route to
            strategy: Load balancing strategy
                - 'round_robin': Rotate through instances
                - 'random': Random selection
                - 'least_loaded': Select least loaded instance
                - 'priority': Use priority ordering

        Returns:
            Service URL or None if not found/healthy
        """
        service = self.services.get(model_name)
        if not service:
            logger.warning(f"⚠️ Model '{model_name}' not found in services")
            return None

        if not self.health_status.get(model_name, False):
            logger.warning(f"⚠️ Service '{model_name}' is not healthy")
            return None

        # For now, simple single-instance routing
        # TODO: Implement real multi-instance load balancing
        self.request_counts[model_name] += 1
        return service["url"]

    def mark_unhealthy(self, model_name: str):
        """
        Mark service as unhealthy

        Args:
            model_name: Model name to mark unhealthy
        """
        self.health_status[model_name] = False
        logger.warning(f"⚠️ Service marked unhealthy: {model_name}")

    def mark_healthy(self, model_name: str):
        """
        Mark service as healthy

        Args:
            model_name: Model name to mark healthy
        """
        self.health_status[model_name] = True
        logger.info(f"✅ Service marked healthy: {model_name}")

    def get_stats(self) -> dict:
        """
        Get load balancer statistics

        Returns:
            Dictionary with stats for each service
        """
        return {
            "health_status": self.health_status.copy(),
            "request_counts": self.request_counts.copy()
        }
