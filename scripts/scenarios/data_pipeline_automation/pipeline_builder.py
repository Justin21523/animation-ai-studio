"""
Data Pipeline Automation - Pipeline Builder

Builds and validates pipeline DAGs from YAML/dict configurations.

Author: Animation AI Studio
Date: 2025-12-03
"""

import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .common import (
    Pipeline,
    PipelineStage,
    PipelineState,
    StageType,
    ExecutionStatus,
    generate_pipeline_id,
    validate_dag,
    topological_sort,
    build_parallel_groups,
    parse_stage_outputs,
)

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """
    Pipeline builder for creating and validating pipelines

    Responsibilities:
    - Load pipeline definitions from YAML or dict
    - Validate DAG structure
    - Resolve stage configurations
    - Register custom stage executors
    - Build execution plans
    """

    def __init__(self):
        """Initialize pipeline builder"""
        self.executor_registry: Dict[str, Type] = {}
        logger.debug("PipelineBuilder initialized")

    def load_from_yaml(self, yaml_path: Path) -> Pipeline:
        """
        Load pipeline from YAML file

        Args:
            yaml_path: Path to YAML pipeline definition

        Returns:
            Pipeline object

        Raises:
            FileNotFoundError: If YAML file not found
            ValueError: If YAML is invalid
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"Pipeline YAML not found: {yaml_path}")

        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")

        logger.info(f"Loaded pipeline from {yaml_path}")
        return self.load_from_dict(config)

    def load_from_dict(self, config: dict) -> Pipeline:
        """
        Load pipeline from dictionary

        Args:
            config: Pipeline configuration dict

        Returns:
            Pipeline object

        Raises:
            ValueError: If config is invalid
        """
        # Validate required fields
        if 'pipeline' not in config:
            raise ValueError("Missing 'pipeline' section in config")

        pipeline_config = config['pipeline']
        required_fields = ['name', 'version']
        for field in required_fields:
            if field not in pipeline_config:
                raise ValueError(f"Missing required field: {field}")

        # Extract pipeline metadata
        name = pipeline_config['name']
        version = pipeline_config['version']
        pipeline_id = pipeline_config.get('id') or generate_pipeline_id(name)

        # Parse stages
        if 'stages' not in config:
            raise ValueError("Missing 'stages' section in config")

        stages = []
        for stage_config in config['stages']:
            stage = self._parse_stage(stage_config)
            stages.append(stage)

        # Validate DAG structure
        if not validate_dag(stages):
            raise ValueError("Invalid DAG: circular dependencies or missing dependencies")

        # Create pipeline
        import time
        pipeline = Pipeline(
            id=pipeline_id,
            name=name,
            version=version,
            stages=stages,
            config=pipeline_config.get('config', {}),
            created_at=time.time(),
            state=PipelineState.PENDING
        )

        logger.info(f"Created pipeline '{name}' with {len(stages)} stages")
        return pipeline

    def _parse_stage(self, stage_config: dict) -> PipelineStage:
        """
        Parse stage configuration

        Args:
            stage_config: Stage configuration dict

        Returns:
            PipelineStage object

        Raises:
            ValueError: If stage config is invalid
        """
        # Validate required fields
        required_fields = ['id', 'type', 'config']
        for field in required_fields:
            if field not in stage_config:
                raise ValueError(f"Stage missing required field: {field}")

        stage_id = stage_config['id']
        stage_type_str = stage_config['type']
        depends_on = stage_config.get('depends_on', [])
        config = stage_config['config']

        # Parse stage type
        try:
            stage_type = StageType(stage_type_str)
        except ValueError:
            # Custom stage type
            stage_type = StageType.CUSTOM
            config['custom_type'] = stage_type_str

        # Create stage
        stage = PipelineStage(
            id=stage_id,
            type=stage_type,
            depends_on=depends_on,
            config=config,
            status=ExecutionStatus.PENDING
        )

        return stage

    def validate_pipeline(self, pipeline: Pipeline) -> bool:
        """
        Validate pipeline structure and configuration

        Args:
            pipeline: Pipeline to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        # Validate DAG
        if not validate_dag(pipeline.stages):
            raise ValueError("Invalid DAG structure")

        # Validate stage dependencies exist
        stage_ids = {s.id for s in pipeline.stages}
        for stage in pipeline.stages:
            for dep in stage.depends_on:
                if dep not in stage_ids:
                    raise ValueError(f"Stage '{stage.id}' depends on unknown stage '{dep}'")

        # Validate no duplicate stage IDs
        if len(stage_ids) != len(pipeline.stages):
            raise ValueError("Duplicate stage IDs found")

        logger.info(f"Pipeline '{pipeline.name}' validation passed")
        return True

    def build_execution_plan(self, pipeline: Pipeline) -> List[List[str]]:
        """
        Build execution plan with parallel stage groups

        Args:
            pipeline: Pipeline to build plan for

        Returns:
            List of stage ID groups that can run in parallel
        """
        # Topologically sort stages
        sorted_stages = topological_sort(pipeline.stages)

        # Update pipeline with sorted stages
        pipeline.stages = sorted_stages

        # Build parallel groups
        groups = build_parallel_groups(sorted_stages)

        logger.info(f"Built execution plan with {len(groups)} parallel groups")
        return groups

    def register_executor(self, stage_type: str, executor_class: Type):
        """
        Register custom stage executor

        Args:
            stage_type: Stage type identifier
            executor_class: Executor class
        """
        self.executor_registry[stage_type] = executor_class
        logger.info(f"Registered executor for stage type '{stage_type}'")

    def resolve_stage_config(self,
                            stage: PipelineStage,
                            completed_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve stage configuration by replacing template variables

        Template variables use {stage_id.output_key} syntax.

        Args:
            stage: Pipeline stage
            completed_outputs: Outputs from completed stages

        Returns:
            Resolved stage configuration
        """
        resolved_config = {}

        for key, value in stage.config.items():
            if isinstance(value, str):
                # Resolve template variables
                resolved_value = parse_stage_outputs(value, completed_outputs)
                resolved_config[key] = resolved_value
            else:
                resolved_config[key] = value

        return resolved_config

    def validate_stage_config(self, stage: PipelineStage) -> bool:
        """
        Validate stage configuration

        Args:
            stage: Pipeline stage

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        # Check required config fields based on stage type
        required_fields = self._get_required_config_fields(stage.type)

        for field in required_fields:
            if field not in stage.config:
                raise ValueError(f"Stage '{stage.id}' missing required config field: {field}")

        return True

    def _get_required_config_fields(self, stage_type: StageType) -> List[str]:
        """
        Get required configuration fields for stage type

        Args:
            stage_type: Stage type

        Returns:
            List of required field names
        """
        # Define required fields per stage type
        required_fields_map = {
            StageType.FRAME_EXTRACTION: ['input_video', 'output_dir'],
            StageType.SEGMENTATION: ['input_dir', 'output_dir'],
            StageType.CLUSTERING: ['input_dir', 'output_dir'],
            StageType.TRAINING_DATA_PREP: ['input_dir', 'output_dir'],
            StageType.CUSTOM: [],  # No validation for custom stages
        }

        return required_fields_map.get(stage_type, [])

    def get_stage_dependencies(self, stage_id: str, pipeline: Pipeline) -> List[str]:
        """
        Get all dependencies for a stage (transitive)

        Args:
            stage_id: Stage identifier
            pipeline: Pipeline

        Returns:
            List of all dependency stage IDs (direct and transitive)
        """
        stage = pipeline.get_stage(stage_id)
        if not stage:
            return []

        dependencies = set()
        visited = set()

        def collect_deps(sid: str):
            if sid in visited:
                return
            visited.add(sid)

            s = pipeline.get_stage(sid)
            if s:
                for dep in s.depends_on:
                    dependencies.add(dep)
                    collect_deps(dep)

        collect_deps(stage_id)
        return list(dependencies)

    def get_parallel_capacity(self, pipeline: Pipeline, max_parallel: int = 4) -> int:
        """
        Calculate maximum parallelism possible for pipeline

        Args:
            pipeline: Pipeline
            max_parallel: Maximum parallel stages allowed

        Returns:
            Actual parallelism capacity (min of max group size and max_parallel)
        """
        groups = build_parallel_groups(pipeline.stages)

        max_group_size = max(len(group) for group in groups) if groups else 1

        return min(max_group_size, max_parallel)

    def save_pipeline_definition(self, pipeline: Pipeline, output_path: Path):
        """
        Save pipeline definition to YAML file

        Args:
            pipeline: Pipeline to save
            output_path: Output YAML path
        """
        config = {
            'pipeline': {
                'id': pipeline.id,
                'name': pipeline.name,
                'version': pipeline.version,
                'config': pipeline.config
            },
            'stages': [
                {
                    'id': stage.id,
                    'type': stage.type.value,
                    'depends_on': stage.depends_on,
                    'config': stage.config
                }
                for stage in pipeline.stages
            ]
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        logger.info(f"Saved pipeline definition to {output_path}")

    def clone_pipeline(self, pipeline: Pipeline, new_name: Optional[str] = None) -> Pipeline:
        """
        Clone pipeline with new ID

        Args:
            pipeline: Pipeline to clone
            new_name: Optional new name

        Returns:
            Cloned pipeline
        """
        import copy
        import time

        cloned = copy.deepcopy(pipeline)
        cloned.id = generate_pipeline_id(new_name or pipeline.name)
        if new_name:
            cloned.name = new_name
        cloned.created_at = time.time()
        cloned.state = PipelineState.PENDING
        cloned.started_at = None
        cloned.completed_at = None

        # Reset stage states
        for stage in cloned.stages:
            stage.status = ExecutionStatus.PENDING
            stage.outputs = {}
            stage.duration = None
            stage.error_message = None
            stage.retry_count = 0
            stage.started_at = None
            stage.completed_at = None

        logger.info(f"Cloned pipeline '{pipeline.name}' as '{cloned.name}'")
        return cloned
