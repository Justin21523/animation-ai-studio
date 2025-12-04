"""
Workflow Executor for Orchestration Layer

DAG-based workflow execution engine with:
- Dependency resolution and parallel execution
- Variable interpolation and context passing
- Retry policies and error handling
- Integration with Event Bus, State Manager, and Module Registry
- Checkpoint/resume capability

Author: Animation AI Studio
Date: 2025-12-02
"""

import asyncio
import logging
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from scripts.orchestration.event_bus import EventBus, Priority
from scripts.orchestration.state_manager import StateManager
from scripts.orchestration.module_registry import (
    ModuleRegistry,
    Task,
    TaskResult
)

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RetryPolicy(Enum):
    """Retry policy for failed tasks"""
    NONE = "none"              # No retries
    IMMEDIATE = "immediate"    # Retry immediately
    EXPONENTIAL = "exponential"  # Exponential backoff


@dataclass
class WorkflowTask:
    """Task definition in workflow"""
    task_id: str
    module: str  # Module name (agent, rag, scenario)
    task_type: str  # Task type for the module
    parameters: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    retry_policy: RetryPolicy = RetryPolicy.NONE
    max_retries: int = 3
    timeout: Optional[float] = None
    condition: Optional[str] = None  # Conditional execution (e.g., "$status == 'success'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "module": self.module,
            "task_type": self.task_type,
            "parameters": self.parameters,
            "depends_on": self.depends_on,
            "retry_policy": self.retry_policy.value,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "condition": self.condition
        }


@dataclass
class WorkflowDefinition:
    """Workflow definition"""
    workflow_id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    global_timeout: Optional[float] = None
    variables: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "tasks": [task.to_dict() for task in self.tasks],
            "global_timeout": self.global_timeout,
            "variables": self.variables
        }


@dataclass
class WorkflowState:
    """Runtime state of workflow execution"""
    workflow_id: str
    status: str
    task_states: Dict[str, TaskStatus] = field(default_factory=dict)
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    retry_counts: Dict[str, int] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for checkpoint"""
        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "task_states": {k: v.value for k, v in self.task_states.items()},
            "task_results": {
                k: v.to_dict() for k, v in self.task_results.items()
            },
            "variables": self.variables,
            "retry_counts": self.retry_counts,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


class WorkflowExecutor:
    """
    Workflow Executor

    Executes DAG-based workflows with:
    - Automatic dependency resolution
    - Parallel task execution (where possible)
    - Variable interpolation (${var} syntax)
    - Conditional execution
    - Retry policies
    - Checkpoint/resume
    - Event publishing

    Example:
        # Define workflow
        workflow = WorkflowDefinition(
            workflow_id="dataset_inspector",
            name="Dataset Quality Inspector",
            description="Inspect dataset quality using Agent + RAG + VLM",
            tasks=[
                WorkflowTask(
                    task_id="analyze_dataset",
                    module="agent",
                    task_type="analyze",
                    parameters={
                        "user_request": "Analyze dataset quality",
                        "enable_rag": True
                    }
                ),
                WorkflowTask(
                    task_id="generate_report",
                    module="scenario",
                    task_type="file",
                    parameters={
                        "scenario": "dataset_builder",
                        "operation": "report",
                        "input": "${analyze_dataset.output}"
                    },
                    depends_on=["analyze_dataset"]
                )
            ]
        )

        # Execute
        executor = WorkflowExecutor(
            event_bus=event_bus,
            state_manager=state_manager,
            module_registry=module_registry
        )

        result = await executor.execute(workflow)
    """

    def __init__(
        self,
        event_bus: EventBus,
        state_manager: StateManager,
        module_registry: ModuleRegistry,
        max_concurrent_tasks: int = 4
    ):
        """
        Initialize Workflow Executor

        Args:
            event_bus: Event bus for publishing events
            state_manager: State manager for checkpointing
            module_registry: Module registry for task execution
            max_concurrent_tasks: Maximum concurrent tasks
        """
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.module_registry = module_registry
        self.max_concurrent_tasks = max_concurrent_tasks

        # Execution state
        self.current_workflow: Optional[WorkflowDefinition] = None
        self.current_state: Optional[WorkflowState] = None

        logger.info(f"WorkflowExecutor initialized (max_concurrent={max_concurrent_tasks})")

    async def execute(
        self,
        workflow: WorkflowDefinition,
        resume: bool = False
    ) -> WorkflowState:
        """
        Execute workflow

        Args:
            workflow: Workflow definition
            resume: Resume from checkpoint if exists

        Returns:
            Final workflow state
        """
        self.current_workflow = workflow
        workflow_id = workflow.workflow_id

        # Try to resume from checkpoint
        if resume:
            checkpoint = await self.state_manager.load_checkpoint(workflow_id)
            if checkpoint:
                logger.info(f"Resuming workflow from checkpoint: {workflow_id}")
                self.current_state = self._restore_state(checkpoint)
            else:
                logger.warning(f"No checkpoint found for workflow: {workflow_id}")
                self.current_state = WorkflowState(
                    workflow_id=workflow_id,
                    status="running",
                    variables=workflow.variables.copy()
                )
        else:
            # Fresh execution
            self.current_state = WorkflowState(
                workflow_id=workflow_id,
                status="running",
                variables=workflow.variables.copy()
            )

            # Initialize task states
            for task in workflow.tasks:
                self.current_state.task_states[task.task_id] = TaskStatus.PENDING
                self.current_state.retry_counts[task.task_id] = 0

        # Publish workflow start event
        await self.event_bus.publish(
            "workflow.started",
            {
                "workflow_id": workflow_id,
                "name": workflow.name,
                "total_tasks": len(workflow.tasks)
            },
            priority=Priority.HIGH
        )

        try:
            # Build dependency graph
            dep_graph = self._build_dependency_graph(workflow.tasks)

            # Execute workflow
            await self._execute_workflow(workflow, dep_graph)

            # Mark as completed
            self.current_state.status = "completed"
            self.current_state.end_time = time.time()

            # Publish workflow completion event
            await self.event_bus.publish(
                "workflow.completed",
                {
                    "workflow_id": workflow_id,
                    "duration": self.current_state.end_time - self.current_state.start_time,
                    "completed_tasks": sum(
                        1 for status in self.current_state.task_states.values()
                        if status == TaskStatus.COMPLETED
                    ),
                    "failed_tasks": sum(
                        1 for status in self.current_state.task_states.values()
                        if status == TaskStatus.FAILED
                    )
                },
                priority=Priority.HIGH
            )

            logger.info(f"Workflow completed: {workflow_id}")

        except Exception as e:
            logger.error(f"Workflow failed: {workflow_id}, error: {e}", exc_info=True)
            self.current_state.status = "failed"
            self.current_state.end_time = time.time()

            # Publish failure event
            await self.event_bus.publish(
                "workflow.failed",
                {
                    "workflow_id": workflow_id,
                    "error": str(e)
                },
                priority=Priority.CRITICAL
            )

        finally:
            # Save final checkpoint
            await self.state_manager.save_checkpoint(
                workflow_id,
                self.current_state.to_dict(),
                description="Final state"
            )

        return self.current_state

    def _build_dependency_graph(
        self,
        tasks: List[WorkflowTask]
    ) -> Dict[str, List[str]]:
        """
        Build dependency graph from tasks

        Args:
            tasks: List of workflow tasks

        Returns:
            Dictionary mapping task_id to list of dependent task_ids
        """
        graph = {}
        all_task_ids = {task.task_id for task in tasks}

        for task in tasks:
            # Validate dependencies
            for dep in task.depends_on:
                if dep not in all_task_ids:
                    raise ValueError(
                        f"Task {task.task_id} depends on unknown task: {dep}"
                    )

            graph[task.task_id] = task.depends_on

        # Check for cycles
        if self._has_cycle(graph):
            raise ValueError("Workflow contains circular dependencies")

        return graph

    def _has_cycle(self, graph: Dict[str, List[str]]) -> bool:
        """Check if graph has cycles (DFS-based)"""
        visited = set()
        rec_stack = set()

        def visit(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if visit(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if visit(node):
                    return True

        return False

    async def _execute_workflow(
        self,
        workflow: WorkflowDefinition,
        dep_graph: Dict[str, List[str]]
    ):
        """
        Execute workflow with dependency resolution

        Args:
            workflow: Workflow definition
            dep_graph: Dependency graph
        """
        tasks_by_id = {task.task_id: task for task in workflow.tasks}
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        async def execute_task_with_deps(task_id: str):
            """Execute task after dependencies complete"""
            task = tasks_by_id[task_id]

            # Wait for dependencies
            deps = dep_graph[task_id]
            if deps:
                # Wait for all dependencies to complete
                while not all(
                    self.current_state.task_states.get(dep) in
                    [TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.FAILED]
                    for dep in deps
                ):
                    await asyncio.sleep(0.1)

                # Check if any dependency failed
                if any(
                    self.current_state.task_states.get(dep) == TaskStatus.FAILED
                    for dep in deps
                ):
                    logger.warning(f"Task {task_id} skipped due to failed dependency")
                    self.current_state.task_states[task_id] = TaskStatus.SKIPPED
                    return

            # Check condition
            if task.condition and not self._evaluate_condition(task.condition):
                logger.info(f"Task {task_id} skipped due to condition: {task.condition}")
                self.current_state.task_states[task_id] = TaskStatus.SKIPPED
                return

            # Execute with semaphore (limit concurrency)
            async with semaphore:
                await self._execute_task(task)

        # Create tasks for all workflow tasks
        execution_tasks = [
            execute_task_with_deps(task_id)
            for task_id in tasks_by_id.keys()
        ]

        # Execute all tasks (respecting dependencies and concurrency limit)
        await asyncio.gather(*execution_tasks)

    async def _execute_task(self, task: WorkflowTask):
        """
        Execute single task with retry logic

        Args:
            task: Workflow task to execute
        """
        task_id = task.task_id

        # Mark as running
        self.current_state.task_states[task_id] = TaskStatus.RUNNING

        # Publish task start event
        await self.event_bus.publish(
            "task.started",
            {"workflow_id": self.current_workflow.workflow_id, "task_id": task_id}
        )

        retry_count = 0
        while retry_count <= task.max_retries:
            try:
                # Interpolate variables in parameters
                interpolated_params = self._interpolate_variables(task.parameters)

                # Create Task for module
                module_task = Task(
                    task_id=task_id,
                    task_type=task.task_type,
                    parameters=interpolated_params,
                    timeout=task.timeout,
                    retry_count=retry_count
                )

                # Execute via module registry
                result = await self.module_registry.execute(task.module, module_task)

                if result.success:
                    # Task succeeded
                    self.current_state.task_states[task_id] = TaskStatus.COMPLETED
                    self.current_state.task_results[task_id] = result

                    # Update variables with result
                    self.current_state.variables[f"{task_id}.output"] = result.output
                    self.current_state.variables[f"{task_id}.success"] = True

                    # Publish success event
                    await self.event_bus.publish(
                        "task.completed",
                        {
                            "workflow_id": self.current_workflow.workflow_id,
                            "task_id": task_id,
                            "execution_time": result.execution_time
                        }
                    )

                    logger.info(f"Task completed: {task_id}")

                    # Save checkpoint after successful task
                    await self.state_manager.save_checkpoint(
                        self.current_workflow.workflow_id,
                        self.current_state.to_dict(),
                        description=f"After task: {task_id}"
                    )

                    return

                else:
                    # Task failed, check retry policy
                    if retry_count < task.max_retries:
                        retry_count += 1
                        self.current_state.retry_counts[task_id] = retry_count

                        await self.event_bus.publish(
                            "task.retry",
                            {
                                "workflow_id": self.current_workflow.workflow_id,
                                "task_id": task_id,
                                "retry_count": retry_count,
                                "error": result.error
                            }
                        )

                        # Apply retry policy
                        if task.retry_policy == RetryPolicy.EXPONENTIAL:
                            delay = min(2 ** retry_count, 60)  # Max 60s
                            logger.info(f"Retrying task {task_id} in {delay}s (attempt {retry_count})")
                            await asyncio.sleep(delay)
                        elif task.retry_policy == RetryPolicy.IMMEDIATE:
                            logger.info(f"Retrying task {task_id} immediately (attempt {retry_count})")
                        else:
                            # No retry
                            break
                    else:
                        # Max retries exhausted
                        break

            except Exception as e:
                logger.error(f"Task execution error: {task_id}, error: {e}", exc_info=True)

                if retry_count < task.max_retries and task.retry_policy != RetryPolicy.NONE:
                    retry_count += 1
                    self.current_state.retry_counts[task_id] = retry_count

                    if task.retry_policy == RetryPolicy.EXPONENTIAL:
                        delay = min(2 ** retry_count, 60)
                        await asyncio.sleep(delay)
                else:
                    break

        # Task failed after all retries
        self.current_state.task_states[task_id] = TaskStatus.FAILED
        self.current_state.variables[f"{task_id}.success"] = False

        # Publish failure event
        await self.event_bus.publish(
            "task.failed",
            {
                "workflow_id": self.current_workflow.workflow_id,
                "task_id": task_id,
                "retry_count": retry_count
            },
            priority=Priority.HIGH
        )

        logger.error(f"Task failed: {task_id} after {retry_count} retries")

    def _interpolate_variables(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpolate variables in parameters

        Supports ${variable} and ${task_id.field} syntax.

        Args:
            params: Parameters with potential variable references

        Returns:
            Parameters with variables interpolated
        """
        def interpolate_value(value: Any) -> Any:
            if isinstance(value, str):
                # Find all ${...} patterns
                pattern = r'\$\{([^}]+)\}'
                matches = re.findall(pattern, value)

                for match in matches:
                    var_value = self._get_variable(match)
                    if var_value is not None:
                        # Replace ${var} with value
                        if value == f"${{{match}}}":
                            # Entire string is variable, return raw value
                            return var_value
                        else:
                            # Partial replacement, convert to string
                            value = value.replace(f"${{{match}}}", str(var_value))

                return value

            elif isinstance(value, dict):
                return {k: interpolate_value(v) for k, v in value.items()}

            elif isinstance(value, list):
                return [interpolate_value(item) for item in value]

            else:
                return value

        return {k: interpolate_value(v) for k, v in params.items()}

    def _get_variable(self, var_name: str) -> Any:
        """
        Get variable value

        Supports both direct variables and dotted paths (task_id.field).

        Args:
            var_name: Variable name or path

        Returns:
            Variable value or None if not found
        """
        # Check direct variable
        if var_name in self.current_state.variables:
            return self.current_state.variables[var_name]

        # Check dotted path (e.g., task_id.output.field)
        parts = var_name.split('.')
        value = self.current_state.variables

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value

    def _evaluate_condition(self, condition: str) -> bool:
        """
        Evaluate condition expression

        Simple condition evaluation supporting comparisons.

        Args:
            condition: Condition string (e.g., "$status == 'success'")

        Returns:
            Boolean result
        """
        try:
            # Interpolate variables
            interpolated = self._interpolate_variables({"_cond": condition})["_cond"]

            # Simple eval (UNSAFE in production - should use proper parser)
            # For now, just check basic patterns
            if "==" in interpolated:
                left, right = interpolated.split("==", 1)
                return left.strip() == right.strip()
            elif "!=" in interpolated:
                left, right = interpolated.split("!=", 1)
                return left.strip() != right.strip()
            else:
                # Treat as boolean variable
                return bool(interpolated)

        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False

    def _restore_state(self, checkpoint: Dict[str, Any]) -> WorkflowState:
        """
        Restore workflow state from checkpoint

        Args:
            checkpoint: Checkpoint dictionary

        Returns:
            Restored WorkflowState
        """
        return WorkflowState(
            workflow_id=checkpoint["workflow_id"],
            status=checkpoint["status"],
            task_states={
                k: TaskStatus(v) for k, v in checkpoint["task_states"].items()
            },
            task_results={
                k: TaskResult(**v) for k, v in checkpoint["task_results"].items()
            },
            variables=checkpoint["variables"],
            retry_counts=checkpoint["retry_counts"],
            start_time=checkpoint["start_time"],
            end_time=checkpoint.get("end_time")
        )

    def get_workflow_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current workflow status

        Returns:
            Status dictionary or None if no workflow is running
        """
        if not self.current_state:
            return None

        completed = sum(
            1 for status in self.current_state.task_states.values()
            if status == TaskStatus.COMPLETED
        )
        failed = sum(
            1 for status in self.current_state.task_states.values()
            if status == TaskStatus.FAILED
        )
        running = sum(
            1 for status in self.current_state.task_states.values()
            if status == TaskStatus.RUNNING
        )
        pending = sum(
            1 for status in self.current_state.task_states.values()
            if status == TaskStatus.PENDING
        )

        return {
            "workflow_id": self.current_state.workflow_id,
            "status": self.current_state.status,
            "tasks": {
                "total": len(self.current_state.task_states),
                "completed": completed,
                "failed": failed,
                "running": running,
                "pending": pending
            },
            "duration": (
                (self.current_state.end_time or time.time()) -
                self.current_state.start_time
            )
        }

    def __repr__(self) -> str:
        if self.current_workflow:
            return f"WorkflowExecutor(workflow={self.current_workflow.workflow_id})"
        return "WorkflowExecutor(idle)"
