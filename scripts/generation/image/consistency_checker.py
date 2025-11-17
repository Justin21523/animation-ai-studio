"""
Character Consistency Checker

Uses ArcFace embeddings to validate character consistency across generated images.
Ensures generated characters match reference images and maintain identity.

Architecture:
- ArcFace face recognition model
- Embedding similarity scoring
- Batch consistency validation
- Reference image management

Author: Animation AI Studio
Date: 2025-11-17
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image
import time

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
except ImportError:
    print("WARNING: insightface not installed. Install with: pip install insightface onnxruntime")
    FaceAnalysis = None


@dataclass
class ConsistencyResult:
    """Result of consistency check"""
    is_consistent: bool
    similarity_score: float
    threshold: float
    reference_embedding: Optional[np.ndarray]
    generated_embedding: Optional[np.ndarray]
    face_detected: bool
    details: Dict[str, Any]


class CharacterConsistencyChecker:
    """
    Character Consistency Checker using ArcFace

    Features:
    - Face detection and alignment
    - ArcFace embedding extraction
    - Cosine similarity scoring
    - Reference image management
    - Batch consistency validation

    Usage:
        checker = CharacterConsistencyChecker()
        result = checker.check_consistency(
            reference_image="data/films/luca/characters/luca_ref.jpg",
            generated_image="outputs/luca_generated.png",
            threshold=0.65
        )
        print(f"Consistent: {result.is_consistent}, Score: {result.similarity_score}")
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "cuda",
        providers: Optional[List[str]] = None
    ):
        """
        Initialize Character Consistency Checker

        Args:
            model_name: InsightFace model name ("buffalo_l", "buffalo_s", etc.)
            device: Device to use (cuda/cpu)
            providers: ONNX providers (defaults to CUDA if available)
        """
        if FaceAnalysis is None:
            raise ImportError(
                "insightface not installed. "
                "Install with: pip install insightface onnxruntime-gpu"
            )

        self.device = device
        self.model_name = model_name

        # Determine ONNX providers
        if providers is None:
            if device == "cuda" and torch.cuda.is_available():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

        print(f"Initializing FaceAnalysis with {model_name}...")
        self.app = FaceAnalysis(
            name=model_name,
            providers=providers
        )
        self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

        print(f"✓ FaceAnalysis initialized (providers: {providers})")

    def extract_embedding(
        self,
        image: Union[str, Image.Image, np.ndarray],
        return_face_info: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Extract ArcFace embedding from image

        Args:
            image: Input image (path, PIL.Image, or numpy array)
            return_face_info: Whether to return face detection info

        Returns:
            Face embedding (512-dim vector) or (embedding, face_info) if return_face_info=True
            Returns None if no face detected
        """
        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Detect faces
        faces = self.app.get(image)

        if len(faces) == 0:
            print("WARNING: No face detected in image")
            if return_face_info:
                return None, {"num_faces": 0}
            return None

        if len(faces) > 1:
            print(f"WARNING: {len(faces)} faces detected, using largest face")

        # Use largest face (by bbox area)
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Extract embedding
        embedding = face.embedding  # 512-dim ArcFace embedding

        if return_face_info:
            face_info = {
                "num_faces": len(faces),
                "bbox": face.bbox.tolist(),
                "det_score": float(face.det_score),
                "landmark": face.kps.tolist() if hasattr(face, 'kps') else None,
                "age": int(face.age) if hasattr(face, 'age') else None,
                "gender": int(face.gender) if hasattr(face, 'gender') else None
            }
            return embedding, face_info

        return embedding

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric ("cosine" or "euclidean")

        Returns:
            Similarity score (0.0-1.0 for cosine, lower is better for euclidean)
        """
        if metric == "cosine":
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            return float(similarity)

        elif metric == "euclidean":
            # Euclidean distance (lower is better)
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(distance)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def check_consistency(
        self,
        reference_image: Union[str, Image.Image, np.ndarray],
        generated_image: Union[str, Image.Image, np.ndarray],
        threshold: float = 0.65,
        metric: str = "cosine"
    ) -> ConsistencyResult:
        """
        Check consistency between reference and generated image

        Args:
            reference_image: Reference character image
            generated_image: Generated image to validate
            threshold: Similarity threshold (0.0-1.0 for cosine)
            metric: Similarity metric ("cosine" or "euclidean")

        Returns:
            ConsistencyResult with validation details
        """
        start_time = time.time()

        # Extract embeddings
        ref_embedding, ref_info = self.extract_embedding(reference_image, return_face_info=True)
        gen_embedding, gen_info = self.extract_embedding(generated_image, return_face_info=True)

        # Check if faces detected
        if ref_embedding is None or gen_embedding is None:
            return ConsistencyResult(
                is_consistent=False,
                similarity_score=0.0,
                threshold=threshold,
                reference_embedding=ref_embedding,
                generated_embedding=gen_embedding,
                face_detected=False,
                details={
                    "error": "Face not detected",
                    "reference_faces": ref_info.get("num_faces", 0) if ref_info else 0,
                    "generated_faces": gen_info.get("num_faces", 0) if gen_info else 0,
                    "processing_time": time.time() - start_time
                }
            )

        # Compute similarity
        similarity = self.compute_similarity(ref_embedding, gen_embedding, metric=metric)

        # Check consistency
        is_consistent = similarity >= threshold if metric == "cosine" else similarity <= threshold

        processing_time = time.time() - start_time

        return ConsistencyResult(
            is_consistent=is_consistent,
            similarity_score=similarity,
            threshold=threshold,
            reference_embedding=ref_embedding,
            generated_embedding=gen_embedding,
            face_detected=True,
            details={
                "metric": metric,
                "reference_info": ref_info,
                "generated_info": gen_info,
                "processing_time": processing_time
            }
        )

    def check_batch_consistency(
        self,
        reference_image: Union[str, Image.Image, np.ndarray],
        generated_images: List[Union[str, Image.Image, np.ndarray]],
        threshold: float = 0.65,
        metric: str = "cosine"
    ) -> List[ConsistencyResult]:
        """
        Check consistency for batch of generated images

        Args:
            reference_image: Reference character image
            generated_images: List of generated images
            threshold: Similarity threshold
            metric: Similarity metric

        Returns:
            List of ConsistencyResult objects
        """
        print(f"Checking consistency for {len(generated_images)} images...")
        results = []

        for i, gen_image in enumerate(generated_images):
            print(f"  [{i+1}/{len(generated_images)}] Processing...", end=" ")
            result = self.check_consistency(
                reference_image=reference_image,
                generated_image=gen_image,
                threshold=threshold,
                metric=metric
            )
            results.append(result)

            status = "✓" if result.is_consistent else "✗"
            print(f"{status} Score: {result.similarity_score:.3f}")

        # Summary
        consistent_count = sum(1 for r in results if r.is_consistent)
        print(f"\nSummary: {consistent_count}/{len(results)} images consistent (threshold: {threshold})")

        return results

    def filter_consistent_images(
        self,
        reference_image: Union[str, Image.Image, np.ndarray],
        generated_images: List[Union[str, Image.Image, np.ndarray]],
        threshold: float = 0.65,
        metric: str = "cosine",
        return_results: bool = False
    ) -> Union[List[Union[str, Image.Image, np.ndarray]], Tuple[List, List[ConsistencyResult]]]:
        """
        Filter generated images to keep only consistent ones

        Args:
            reference_image: Reference character image
            generated_images: List of generated images
            threshold: Similarity threshold
            metric: Similarity metric
            return_results: Whether to also return ConsistencyResult objects

        Returns:
            List of consistent images, or (images, results) if return_results=True
        """
        results = self.check_batch_consistency(
            reference_image=reference_image,
            generated_images=generated_images,
            threshold=threshold,
            metric=metric
        )

        consistent_images = [
            img for img, result in zip(generated_images, results)
            if result.is_consistent
        ]

        if return_results:
            consistent_results = [r for r in results if r.is_consistent]
            return consistent_images, consistent_results

        return consistent_images

    def compute_average_embedding(
        self,
        images: List[Union[str, Image.Image, np.ndarray]]
    ) -> Optional[np.ndarray]:
        """
        Compute average embedding from multiple images of same character

        Useful for creating robust reference embeddings

        Args:
            images: List of character images

        Returns:
            Average embedding (512-dim) or None if no faces detected
        """
        embeddings = []

        for image in images:
            embedding = self.extract_embedding(image)
            if embedding is not None:
                embeddings.append(embedding)

        if len(embeddings) == 0:
            print("WARNING: No faces detected in any image")
            return None

        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)

        # Normalize (important for cosine similarity)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        print(f"✓ Computed average embedding from {len(embeddings)} images")
        return avg_embedding

    def save_embedding(self, embedding: np.ndarray, save_path: str):
        """Save embedding to disk"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, embedding)
        print(f"✓ Embedding saved to: {save_path}")

    def load_embedding(self, load_path: str) -> np.ndarray:
        """Load embedding from disk"""
        embedding = np.load(load_path)
        print(f"✓ Embedding loaded from: {load_path}")
        return embedding


class CharacterReferenceManager:
    """
    Manages reference embeddings for characters

    Features:
    - Store reference embeddings for each character
    - Load from pre-computed embeddings
    - Compute from multiple reference images
    """

    def __init__(
        self,
        consistency_checker: CharacterConsistencyChecker,
        embeddings_dir: str = "data/character_embeddings"
    ):
        """
        Initialize Character Reference Manager

        Args:
            consistency_checker: CharacterConsistencyChecker instance
            embeddings_dir: Directory to store character embeddings
        """
        self.checker = consistency_checker
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self.character_embeddings: Dict[str, np.ndarray] = {}

    def get_embedding_path(self, character_name: str) -> Path:
        """Get path for character embedding file"""
        return self.embeddings_dir / f"{character_name}_embedding.npy"

    def has_embedding(self, character_name: str) -> bool:
        """Check if embedding exists for character"""
        return self.get_embedding_path(character_name).exists()

    def load_character_embedding(self, character_name: str) -> Optional[np.ndarray]:
        """
        Load character embedding

        Args:
            character_name: Character name

        Returns:
            Character embedding or None if not found
        """
        if character_name in self.character_embeddings:
            return self.character_embeddings[character_name]

        embedding_path = self.get_embedding_path(character_name)
        if not embedding_path.exists():
            print(f"No embedding found for character: {character_name}")
            return None

        embedding = self.checker.load_embedding(str(embedding_path))
        self.character_embeddings[character_name] = embedding
        return embedding

    def create_character_embedding(
        self,
        character_name: str,
        reference_images: List[Union[str, Image.Image, np.ndarray]],
        save: bool = True
    ) -> np.ndarray:
        """
        Create character embedding from reference images

        Args:
            character_name: Character name
            reference_images: List of reference images
            save: Whether to save embedding to disk

        Returns:
            Character embedding
        """
        print(f"Creating embedding for character: {character_name}")

        # Compute average embedding
        embedding = self.checker.compute_average_embedding(reference_images)

        if embedding is None:
            raise ValueError(f"Failed to create embedding for {character_name}")

        # Store in memory
        self.character_embeddings[character_name] = embedding

        # Save to disk
        if save:
            self.checker.save_embedding(embedding, str(self.get_embedding_path(character_name)))

        return embedding

    def check_character_consistency(
        self,
        character_name: str,
        generated_image: Union[str, Image.Image, np.ndarray],
        threshold: float = 0.65
    ) -> ConsistencyResult:
        """
        Check consistency for character using stored embedding

        Args:
            character_name: Character name
            generated_image: Generated image
            threshold: Similarity threshold

        Returns:
            ConsistencyResult
        """
        # Load character embedding
        ref_embedding = self.load_character_embedding(character_name)
        if ref_embedding is None:
            raise ValueError(f"No reference embedding for character: {character_name}")

        # Extract generated image embedding
        gen_embedding, gen_info = self.checker.extract_embedding(generated_image, return_face_info=True)

        if gen_embedding is None:
            return ConsistencyResult(
                is_consistent=False,
                similarity_score=0.0,
                threshold=threshold,
                reference_embedding=ref_embedding,
                generated_embedding=None,
                face_detected=False,
                details={"error": "No face detected in generated image"}
            )

        # Compute similarity
        similarity = self.checker.compute_similarity(ref_embedding, gen_embedding)
        is_consistent = similarity >= threshold

        return ConsistencyResult(
            is_consistent=is_consistent,
            similarity_score=similarity,
            threshold=threshold,
            reference_embedding=ref_embedding,
            generated_embedding=gen_embedding,
            face_detected=True,
            details={"generated_info": gen_info, "character": character_name}
        )


def main():
    """Example usage"""

    # Initialize consistency checker
    checker = CharacterConsistencyChecker(
        model_name="buffalo_l",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Example 1: Check consistency between two images
    print("\n=== Example 1: Check Consistency ===")
    reference_image = "data/films/luca/characters/luca_ref.jpg"
    generated_image = "outputs/luca_generated.png"

    # Note: These paths are examples, actual files may not exist
    if Path(reference_image).exists() and Path(generated_image).exists():
        result = checker.check_consistency(
            reference_image=reference_image,
            generated_image=generated_image,
            threshold=0.65
        )

        print(f"Consistent: {result.is_consistent}")
        print(f"Similarity: {result.similarity_score:.3f}")
        print(f"Threshold: {result.threshold}")

    # Example 2: Use reference manager
    print("\n=== Example 2: Reference Manager ===")
    ref_manager = CharacterReferenceManager(checker)

    # Create character embedding from multiple references
    reference_images = [
        "data/films/luca/characters/luca_ref_1.jpg",
        "data/films/luca/characters/luca_ref_2.jpg",
        "data/films/luca/characters/luca_ref_3.jpg"
    ]

    # Note: This is an example, actual files may not exist
    existing_refs = [img for img in reference_images if Path(img).exists()]
    if existing_refs:
        ref_manager.create_character_embedding(
            character_name="luca",
            reference_images=existing_refs,
            save=True
        )

        # Check consistency using stored embedding
        if Path(generated_image).exists():
            result = ref_manager.check_character_consistency(
                character_name="luca",
                generated_image=generated_image,
                threshold=0.65
            )
            print(f"Character consistency: {result.is_consistent} (score: {result.similarity_score:.3f})")

    print("\n✓ Examples complete!")


if __name__ == "__main__":
    main()
