"""
Module Registry for Orchestration Layer

Central registry for all automation modules and their adapters.
Provides standardized interface for module discovery, health checking, and execution.

Usage:
    from scripts.orchestration.module_registry import ModuleRegistry, ModuleAdapter

    # Create adapter
    class MyAdapter(ModuleAdapter):
        async def execute(self, task):
            # Implementation
            pass

    # Register module
    registry = ModuleRegistry()
    registry.register("my_module", MyAdapter(...))

    # Execute task
    result = await registry.execute("my_module", task)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import time

logger = logging.getLogger(__name__)


class ModuleType(Enum):
    """Module type classification."""
    AGENT = "agent"
    RAG = "rag"
    VLM = "vlm"
    SCENARIO = "scenario"
    CUSTOM = "custom"


class ModuleStatus(Enum):
    """Module health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class Task:
    """Task data structure for module execution."""
    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    timeout: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModuleCapabilities:
    """Module capability description."""
    module_name: str
    module_type: ModuleType
    supported_operations: List[str]
    requires_gpu: bool = False
    max_concurrent_tasks: int = 1
    estimated_memory_mb: int = 0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["module_type"] = self.module_type.value
        return result


@dataclass
class HealthCheckResult:
    """Health check result."""
    module_name: str
    status: ModuleStatus
    timestamp: float = field(default_factory=time.time)
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["status"] = self.status.value
        return result


class ModuleAdapter(ABC):
    """
    Base adapter interface for all modules.

    All module adapters must inherit from this class and implement
    the abstract methods. This provides a standardized interface for
    the orchestration layer to interact with different module types.
    """

    def __init__(self, module_name: str, module_type: ModuleType):
        """
        Initialize module adapter.

        Args:
            module_name: Unique module name
            module_type: Module type classification
        """
        self.module_name = module_name
        self.module_type = module_type
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the module.

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    async def execute(self, task: Task) -> TaskResult:
        """
        Execute task using this module.

        Args:
            task: Task to execute

        Returns:
            Task execution result
        """
        pass

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """
        Check module health status.

        Returns:
            Health check result
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> ModuleCapabilities:
        """
        Get module capabilities.

        Returns:
            Module capabilities description
        """
        pass

    async def cleanup(self):
        """
        Cleanup module resources.

        Optional method for resource cleanup.
        """
        pass

    def is_initialized(self) -> bool:
        """Check if module is initialized."""
        return self._initialized


class ModuleRegistry:
    """
    Central registry for all automation modules.

    Features:
    - Module registration and discovery
    - Health monitoring
    - Capability querying
    - Standardized task execution interface

    Example:
        registry = ModuleRegistry()

        # Register modules
        registry.register("agent", AgentAdapter(...))
        registry.register("rag", RAGAdapter(...))

        # Execute task
        task = Task(task_id="1", task_type="analyze", parameters={...})
        result = await registry.execute("agent", task)

        # Health check
        health = await registry.health_check_all()
    """

    def __init__(self):
        """Initialize module registry."""
        self._modules: Dict[str, ModuleAdapter] = {}
        self._initialized_modules: Set[str] = set()

        logger.info("ModuleRegistry initialized")

    def register(self, module_name: str, adapter: ModuleAdapter):
        """
        Register module adapter.

        Args:
            module_name: Unique module name
            adapter: Module adapter instance

        Raises:
            ValueError: If module already registered
        """
        if module_name in self._modules:
            raise ValueError(f"Module already registered: {module_name}")

        self._modules[module_name] = adapter
        logger.info(f"Registered module: {module_name}, type={adapter.module_type.value}")

    def unregister(self, module_name: str) -> bool:
        """
        Unregister module.

        Args:
            module_name: Module name to unregister

        Returns:
            True if module was unregistered
        """
        if module_name in self._modules:
            del self._modules[module_name]
            self._initialized_modules.discard(module_name)
            logger.info(f"Unregistered module: {module_name}")
            return True
        return False

    def get_module(self, module_name: str) -> Optional[ModuleAdapter]:
        """
        Get module adapter by name.

        Args:
            module_name: Module name

        Returns:
            Module adapter or None if not found
        """
        return self._modules.get(module_name)

    def list_modules(self, module_type: Optional[ModuleType] = None) -> List[str]:
        """
        List all registered modules.

        Args:
            module_type: Optional filter by module type

        Returns:
            List of module names
        """
        if module_type is None:
            return list(self._modules.keys())

        return [
            name for name, adapter in self._modules.items()
            if adapter.module_type == module_type
        ]

    async def initialize_module(self, module_name: str) -> bool:
        """
        Initialize specific module.

        Args:
            module_name: Module name to initialize

        Returns:
            True if initialization successful

        Raises:
            KeyError: If module not found
        """
        if module_name not in self._modules:
            raise KeyError(f"Module not found: {module_name}")

        if module_name in self._initialized_modules:
            logger.debug(f"Module already initialized: {module_name}")
            return True

        adapter = self._modules[module_name]
        success = await adapter.initialize()

        if success:
            self._initialized_modules.add(module_name)
            logger.info(f"Initialized module: {module_name}")
        else:
            logger.error(f"Failed to initialize module: {module_name}")

        return success

    async def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all registered modules.

        Returns:
            Dictionary mapping module names to initialization success
        """
        results = {}
        for module_name in self._modules.keys():
            try:
                results[module_name] = await self.initialize_module(module_name)
            except Exception as e:
                logger.error(f"Error initializing {module_name}: {e}", exc_info=True)
                results[module_name] = False

        logger.info(f"Initialized {sum(results.values())}/{len(results)} modules")
        return results

    async def execute(self, module_name: str, task: Task) -> TaskResult:
        """
        Execute task using specified module.

        Args:
            module_name: Module name to execute task with
            task: Task to execute

        Returns:
            Task execution result

        Raises:
            KeyError: If module not found
            RuntimeError: If module not initialized
        """
        if module_name not in self._modules:
            raise KeyError(f"Module not found: {module_name}")

        if module_name not in self._initialized_modules:
            raise RuntimeError(f"Module not initialized: {module_name}")

        adapter = self._modules[module_name]
        start_time = time.time()

        try:
            result = await adapter.execute(task)
            result.execution_time = time.time() - start_time
            logger.debug(
                f"Task executed: module={module_name}, task_id={task.task_id}, "
                f"success={result.success}, time={result.execution_time:.2f}s"
            )
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task execution failed: module={module_name}, error={e}", exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    async def health_check(self, module_name: str) -> HealthCheckResult:
        """
        Check health of specific module.

        Args:
            module_name: Module name to check

        Returns:
            Health check result

        Raises:
            KeyError: If module not found
        """
        if module_name not in self._modules:
            raise KeyError(f"Module not found: {module_name}")

        adapter = self._modules[module_name]

        try:
            result = await adapter.health_check()
            logger.debug(f"Health check: module={module_name}, status={result.status.value}")
            return result

        except Exception as e:
            logger.error(f"Health check failed: module={module_name}, error={e}", exc_info=True)
            return HealthCheckResult(
                module_name=module_name,
                status=ModuleStatus.UNHEALTHY,
                message=f"Health check error: {str(e)}"
            )

    async def health_check_all(self) -> Dict[str, HealthCheckResult]:
        """
        Check health of all registered modules.

        Returns:
            Dictionary mapping module names to health check results
        """
        results = {}
        for module_name in self._modules.keys():
            try:
                results[module_name] = await self.health_check(module_name)
            except Exception as e:
                logger.error(f"Health check error for {module_name}: {e}", exc_info=True)
                results[module_name] = HealthCheckResult(
                    module_name=module_name,
                    status=ModuleStatus.UNKNOWN,
                    message=str(e)
                )

        return results

    def get_capabilities(self, module_name: str) -> ModuleCapabilities:
        """
        Get capabilities of specific module.

        Args:
            module_name: Module name

        Returns:
            Module capabilities

        Raises:
            KeyError: If module not found
        """
        if module_name not in self._modules:
            raise KeyError(f"Module not found: {module_name}")

        adapter = self._modules[module_name]
        return adapter.get_capabilities()

    def get_all_capabilities(self) -> Dict[str, ModuleCapabilities]:
        """
        Get capabilities of all modules.

        Returns:
            Dictionary mapping module names to capabilities
        """
        return {
            module_name: adapter.get_capabilities()
            for module_name, adapter in self._modules.items()
        }

    async def cleanup_module(self, module_name: str):
        """
        Cleanup specific module resources.

        Args:
            module_name: Module name to cleanup

        Raises:
            KeyError: If module not found
        """
        if module_name not in self._modules:
            raise KeyError(f"Module not found: {module_name}")

        adapter = self._modules[module_name]
        await adapter.cleanup()
        self._initialized_modules.discard(module_name)
        logger.info(f"Cleaned up module: {module_name}")

    async def cleanup_all(self):
        """Cleanup all module resources."""
        for module_name in list(self._modules.keys()):
            try:
                await self.cleanup_module(module_name)
            except Exception as e:
                logger.error(f"Error cleaning up {module_name}: {e}", exc_info=True)

        logger.info("All modules cleaned up")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_modules": len(self._modules),
            "initialized_modules": len(self._initialized_modules),
            "modules_by_type": {
                module_type.value: len(self.list_modules(module_type))
                for module_type in ModuleType
            }
        }

    def __repr__(self) -> str:
        return f"ModuleRegistry(modules={len(self._modules)}, initialized={len(self._initialized_modules)})"
