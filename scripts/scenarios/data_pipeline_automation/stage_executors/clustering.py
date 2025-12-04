"""
Data Pipeline Automation - CLIP Character Clustering Executor

Wraps clip_character_clustering.py for pipeline execution.

Author: Animation AI Studio
Date: 2025-12-04
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from .script_executor import ScriptExecutor
from ..common import StageResult, ExecutionStatus


class ClusteringExecutor(ScriptExecutor):
    """
    CLIP-based character clustering stage executor

    Wraps clip_character_clustering.py to cluster character instances
    by identity using CLIP visual embeddings.

    Required config keys:
        - instances_dir: Directory containing character instance images

    Optional config keys:
        - output_dir: Output directory for clusters (required if project not specified)
        - project: Project/film name (auto-constructs paths)
        - method: Clustering method (default: kmeans)
            - kmeans: Fixed k or auto k detection (100% coverage)
            - hdbscan: Density-based (may produce noise cluster)
        - k_range: K-means k range for auto detection (default: [10, 50])
        - fixed_k: Manually specify k for K-means (overrides auto detection)
        - min_cluster_size: HDBSCAN minimum instances per cluster (default: 12)
        - min_samples: HDBSCAN minimum samples for core points (default: 2)
        - batch_size: Batch size for CLIP encoding (default: 64)
        - device: Device to use (default: cuda)

    Outputs:
        - output_dir: Root output directory
        - cluster_dirs: List of cluster directories (cluster_0, cluster_1, ...)
        - cluster_count: Number of clusters found
        - total_instances: Total instances clustered
        - noise_count: Number of noise instances (HDBSCAN only)
        - silhouette_score: Clustering quality metric

    Metrics:
        - instances_clustered: Total instances
        - characters_found: Number of clusters (characters)
        - noise_instances: Noise count (HDBSCAN)
        - silhouette_score: Quality metric (-1 to 1, higher is better)
        - clustering_time: Execution time in seconds
    """

    # Script configuration
    script_path = "scripts/processing/clustering/clip_character_clustering.py"
    required_config_keys = ["instances_dir"]
    output_keys = ["output_dir", "cluster_dirs", "cluster_count", "total_instances", "noise_count", "silhouette_score"]

    def validate_config(self) -> bool:
        """
        Validate clustering configuration

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Call parent validation
        super().validate_config()

        # Validate instances directory exists
        instances_dir = Path(self._get_config_value("instances_dir", required=True))
        self._validate_path_exists(instances_dir, "Instances directory")

        # Check for image files in instances directory
        image_count = self._count_files(instances_dir, "*.png")
        image_count += self._count_files(instances_dir, "*.jpg")
        image_count += self._count_files(instances_dir, "*.jpeg")

        if image_count == 0:
            raise ValueError(f"No image files found in instances directory: {instances_dir}")

        # Need at least 10 images for meaningful clustering
        if image_count < 10:
            raise ValueError(f"Too few instances for clustering (< 10): {image_count}")

        # Validate output_dir or project is specified
        output_dir = self._get_config_value("output_dir")
        project = self._get_config_value("project")
        if not output_dir and not project:
            raise ValueError("Either 'output_dir' or 'project' must be specified")

        # Validate method
        method = self._get_config_value("method", default="kmeans")
        if method not in ["kmeans", "hdbscan"]:
            raise ValueError(f"Invalid method '{method}'. Must be 'kmeans' or 'hdbscan'")

        # Validate K-means parameters
        if method == "kmeans":
            k_range = self._get_config_value("k_range", default=[10, 50])
            if not isinstance(k_range, list) or len(k_range) != 2:
                raise ValueError(f"k_range must be a list of two integers, got {k_range}")

            min_k, max_k = k_range
            if min_k < 2:
                raise ValueError(f"Minimum k must be >= 2, got {min_k}")
            if max_k <= min_k:
                raise ValueError(f"Maximum k must be > minimum k, got {min_k}-{max_k}")

            fixed_k = self._get_config_value("fixed_k")
            if fixed_k is not None and fixed_k < 2:
                raise ValueError(f"Fixed k must be >= 2, got {fixed_k}")

        # Validate HDBSCAN parameters
        if method == "hdbscan":
            min_cluster_size = self._get_config_value("min_cluster_size", default=12)
            if min_cluster_size < 2:
                raise ValueError(f"min_cluster_size must be >= 2, got {min_cluster_size}")

            min_samples = self._get_config_value("min_samples", default=2)
            if min_samples < 1:
                raise ValueError(f"min_samples must be >= 1, got {min_samples}")

        # Validate batch size
        batch_size = self._get_config_value("batch_size", default=64)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        # Validate device
        device = self._get_config_value("device", default="cuda")
        if device not in ["cuda", "cpu"]:
            raise ValueError(f"Device must be 'cuda' or 'cpu', got '{device}'")

        return True

    def _build_command(self, inputs: Dict[str, Any]) -> List[str]:
        """
        Build command to execute clustering script

        Args:
            inputs: Input data from previous stages
                   Can contain 'character_dir' from segmentation stage

        Returns:
            Command line arguments list
        """
        # Start with base command
        command = ["python", self.script_path]

        # Add required positional argument (instances directory)
        # Check if instances_dir comes from previous stage output
        instances_dir = self.config.get("instances_dir")
        if instances_dir and isinstance(instances_dir, str) and "{" in instances_dir:
            # Template string like "{segment_characters.character_dir}"
            from ..common import parse_stage_outputs
            instances_dir = parse_stage_outputs(instances_dir, inputs)
        else:
            instances_dir = self._get_config_value("instances_dir", required=True)

        command.append(str(instances_dir))

        # Add output directory (if specified)
        output_dir = self._get_config_value("output_dir")
        if output_dir:
            command.extend(["--output-dir", str(output_dir)])

        # Add project name (if specified)
        project = self._get_config_value("project")
        if project:
            command.extend(["--project", str(project)])

        # Add method
        method = self._get_config_value("method", default="kmeans")
        command.extend(["--method", method])

        # Add method-specific parameters
        if method == "kmeans":
            k_range = self._get_config_value("k_range", default=[10, 50])
            command.extend(["--k-range", str(k_range[0]), str(k_range[1])])

            fixed_k = self._get_config_value("fixed_k")
            if fixed_k is not None:
                command.extend(["--fixed-k", str(fixed_k)])

        else:  # hdbscan
            min_cluster_size = self._get_config_value("min_cluster_size", default=12)
            command.extend(["--min-cluster-size", str(min_cluster_size)])

            min_samples = self._get_config_value("min_samples", default=2)
            command.extend(["--min-samples", str(min_samples)])

        # Add batch size
        batch_size = self._get_config_value("batch_size", default=64)
        command.extend(["--batch-size", str(batch_size)])

        # Add device
        device = self._get_config_value("device", default="cuda")
        command.extend(["--device", device])

        return command

    def _parse_outputs(self) -> Dict[str, Any]:
        """
        Parse clustering outputs

        Returns:
            Dictionary containing:
                - output_dir: Root output directory
                - cluster_dirs: List of cluster directory paths
                - cluster_count: Number of clusters
                - total_instances: Total instances
                - noise_count: Noise instances (HDBSCAN)
                - silhouette_score: Quality metric

        Raises:
            RuntimeError: If output parsing fails
        """
        outputs = {}

        # Determine output directory
        output_dir = self._get_config_value("output_dir")
        project = self._get_config_value("project")

        if project and not output_dir:
            # Auto-constructed path
            base_dir = Path("/mnt/data/ai_data/datasets/3d-anime")
            output_path = base_dir / project / "clustered"
        elif output_dir:
            output_path = Path(output_dir)
        else:
            raise RuntimeError("Cannot determine output directory")

        # Validate output directory exists
        if not output_path.exists():
            raise RuntimeError(f"Output directory not found: {output_path}")

        outputs["output_dir"] = str(output_path)

        # Load clustering_metadata.json
        metadata_file = output_path / "clustering_metadata.json"
        if metadata_file.exists():
            try:
                metadata = self._load_json_file(metadata_file)

                # Extract clustering info
                clustering_info = metadata.get("clustering_info", {})
                outputs["cluster_count"] = clustering_info.get("n_clusters", 0)
                outputs["total_instances"] = metadata.get("total_instances", 0)
                outputs["noise_count"] = clustering_info.get("n_noise", 0)
                outputs["silhouette_score"] = clustering_info.get("silhouette_score", 0.0)

                # Store full info for metrics
                outputs["_clustering_info"] = clustering_info

            except Exception as e:
                raise RuntimeError(f"Failed to parse clustering_metadata.json: {e}")
        else:
            # Fallback: count cluster directories manually
            cluster_dirs = sorted(output_path.glob("cluster_*"))
            outputs["cluster_count"] = len(cluster_dirs)
            outputs["total_instances"] = 0
            outputs["noise_count"] = 0
            outputs["silhouette_score"] = 0.0
            outputs["_clustering_info"] = {}

        # Find all cluster directories
        cluster_dirs = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("cluster_")])
        outputs["cluster_dirs"] = [str(d) for d in cluster_dirs]

        return outputs

    def _extract_metrics(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract performance metrics

        Args:
            outputs: Parsed outputs from _parse_outputs()

        Returns:
            Dictionary containing:
                - instances_clustered: Total instances
                - characters_found: Number of clusters
                - noise_instances: Noise count
                - silhouette_score: Quality metric
                - clustering_time: Execution time
        """
        metrics = {}

        # Instance count
        total_instances = outputs.get("total_instances", 0)
        metrics["instances_clustered"] = float(total_instances)

        # Number of characters (clusters)
        cluster_count = outputs.get("cluster_count", 0)
        metrics["characters_found"] = float(cluster_count)

        # Noise instances
        noise_count = outputs.get("noise_count", 0)
        metrics["noise_instances"] = float(noise_count)

        # Quality metric
        silhouette = outputs.get("silhouette_score", 0.0)
        metrics["silhouette_score"] = float(silhouette)

        # Extract additional metrics from clustering_info
        clustering_info = outputs.get("_clustering_info", {})
        if "davies_bouldin_score" in clustering_info:
            metrics["davies_bouldin_score"] = float(clustering_info["davies_bouldin_score"])

        # Calculate clustering time
        if self.start_time and self.end_time:
            clustering_time = self.end_time - self.start_time
            metrics["clustering_time"] = clustering_time
        else:
            metrics["clustering_time"] = 0.0

        return metrics

    def estimate_duration(self) -> float:
        """
        Estimate clustering duration

        Returns:
            Estimated duration in seconds

        Notes:
            CLIP embedding extraction is the main bottleneck:
            - CUDA: ~0.05-0.1 sec/image
            - CPU: ~0.3-0.5 sec/image
            Clustering itself is fast (<10 seconds for < 10k instances)
        """
        device = self._get_config_value("device", default="cuda")
        batch_size = self._get_config_value("batch_size", default=64)

        # Time per image estimates (seconds)
        time_per_image = 0.075 if device == "cuda" else 0.4

        # Try to count images in instances directory
        try:
            instances_dir = Path(self._get_config_value("instances_dir", required=True))
            image_count = self._count_files(instances_dir, "*.png")
            image_count += self._count_files(instances_dir, "*.jpg")
            image_count += self._count_files(instances_dir, "*.jpeg")

            if image_count > 0:
                # Estimate total duration for embedding extraction
                embedding_time = image_count * time_per_image

                # Add clustering overhead (typically <10 seconds, but depends on k)
                clustering_overhead = 10.0

                # Add batch processing efficiency (~80% efficient)
                total_time = (embedding_time * 1.2) + clustering_overhead

                return total_time
        except Exception:
            pass

        # Default: 10 minutes
        return 600.0
