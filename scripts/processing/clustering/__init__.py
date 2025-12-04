"""
Clustering Modules
Provides character instance clustering tools.

Multi-Stage Clustering Pipeline:
1. Instance Pre-filtering: Remove background/low-quality instances
2. Identity Clustering: Group by character identity (ArcFace or CLIP)
3. Pose/View Subclustering: Subdivide by pose and viewing angle (optional)
4. Interactive Review: Manual refinement and merging

Components:
- clip_character_clustering: CLIP-based visual similarity clustering
- face_identity_clustering: ArcFace-based identity clustering (RECOMMENDED for multi-character)
- action_clustering: Action/activity-based clustering
- pose_subclustering: Pose and view angle subclustering
- scene_clustering: Scene-level clustering
- instance_prefilter: Pre-filtering to remove background instances
- cluster_organizer: Cluster organization utilities
- cluster_review_and_merge: Interactive cluster review interface
- visualize_scene_clustering: Visualization tools
"""

# Individual scripts are standalone CLI tools

__all__ = [
    # Main modules are used as CLI scripts
]
