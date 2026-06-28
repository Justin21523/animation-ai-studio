"""
Results Browser API - Browse and view job output files.

Provides endpoints for:
- Browsing directory tree
- Getting file metadata
- Listing job outputs
"""
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel


# Pydantic models
class FileInfo(BaseModel):
    """File or directory information."""
    name: str
    path: str
    type: str  # 'file' or 'directory'
    size: Optional[int] = None
    modified: Optional[str] = None
    extension: Optional[str] = None


class DirectoryListing(BaseModel):
    """Directory listing response."""
    current_path: str
    parent_path: Optional[str]
    items: List[FileInfo]
    total_files: int
    total_directories: int


# Create router
router = APIRouter(prefix="/api/results", tags=["results"])


# Allowed base directories (security constraint). main.py updates this from config.
ALLOWED_BASE_DIRS = [
    str((Path(__file__).resolve().parents[3] / "outputs" / "demo").resolve()),
]


def set_allowed_base_dirs(paths: List[str]) -> None:
    """Set result browser roots from backend configuration."""
    global ALLOWED_BASE_DIRS
    cleaned = [str(Path(path).resolve()) for path in paths if path]
    if cleaned:
        ALLOWED_BASE_DIRS = cleaned


def is_path_allowed(path: str) -> bool:
    """
    Check if path is within allowed base directories.

    Security measure to prevent directory traversal attacks.
    """
    try:
        abs_path = Path(path).resolve()
    except Exception:
        abs_path = Path(os.path.abspath(path))

    for base in ALLOWED_BASE_DIRS:
        try:
            base_path = Path(base).resolve()
            abs_path.relative_to(base_path)
            return True
        except Exception:
            continue
    return False


def get_file_info(file_path: Path) -> FileInfo:
    """
    Get metadata for a file or directory.

    Args:
        file_path: Path to file or directory

    Returns:
        FileInfo object with metadata
    """
    stat = file_path.stat()

    return FileInfo(
        name=file_path.name,
        path=str(file_path),
        type="directory" if file_path.is_dir() else "file",
        size=stat.st_size if file_path.is_file() else None,
        modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        extension=file_path.suffix if file_path.is_file() else None
    )


@router.get("/", response_model=DirectoryListing)
async def browse_directory(
    path: Optional[str] = Query(None, description="Directory path to browse")
):
    """
    Browse a directory and list its contents.

    Query Parameters:
        path: Directory path to browse (must be within allowed base dirs)

    Returns:
        DirectoryListing with files and subdirectories

    Example:
        GET /api/results/?path=/mnt/data/extracted/film_name
    """
    if path is None:
        path = ALLOWED_BASE_DIRS[0]

    # Security check
    if not is_path_allowed(path):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Path must be within allowed directories: {ALLOWED_BASE_DIRS}"
        )

    dir_path = Path(path)

    # The default result root is allowed to be empty on a fresh demo install.
    if not dir_path.exists() and str(dir_path.resolve()) in ALLOWED_BASE_DIRS:
        dir_path.mkdir(parents=True, exist_ok=True)

    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")

    if not dir_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Path is not a directory: {path}"
        )

    # List directory contents
    items = []
    total_files = 0
    total_directories = 0

    try:
        for item in sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
            try:
                file_info = get_file_info(item)
                items.append(file_info)

                if item.is_dir():
                    total_directories += 1
                else:
                    total_files += 1
            except (PermissionError, OSError):
                # Skip files we can't read
                continue
    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail=f"Permission denied: {path}"
        )

    # Get parent directory
    parent_path = str(dir_path.parent) if dir_path.parent != dir_path else None

    return DirectoryListing(
        current_path=str(dir_path),
        parent_path=parent_path,
        items=items,
        total_files=total_files,
        total_directories=total_directories
    )


@router.get("/browse")
async def browse_directory_compat(
    path: Optional[str] = Query(None, description="Directory path to browse")
):
    """
    Compatibility endpoint for the frontend ResultBrowser.

    Returns:
        {"files": [...], "current_path": "...", "parent_path": "..."}
    """
    listing = await browse_directory(path=path)
    return {
        "current_path": listing.current_path,
        "parent_path": listing.parent_path,
        "files": [item.model_dump() for item in listing.items],
        "total_files": listing.total_files,
        "total_directories": listing.total_directories,
    }


@router.get("/file", response_model=FileInfo)
async def get_file_metadata(
    path: str = Query(..., description="File path")
):
    """
    Get metadata for a specific file.

    Query Parameters:
        path: File path

    Returns:
        FileInfo with file metadata

    Example:
        GET /api/results/file?path=/mnt/data/extracted/film/frames/frame_0001.jpg
    """
    # Security check
    if not is_path_allowed(path):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Path must be within allowed directories: {ALLOWED_BASE_DIRS}"
        )

    file_path = Path(path)

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {path}"
        )

    try:
        return get_file_info(file_path)
    except (PermissionError, OSError) as e:
        raise HTTPException(
            status_code=403,
            detail=f"Cannot access file: {str(e)}"
        )


@router.get("/metadata", response_model=FileInfo)
async def get_file_metadata_compat(path: str = Query(..., description="File path")):
    """Compatibility endpoint for the frontend ResultBrowser."""
    return await get_file_metadata(path=path)


@router.get("/download")
async def download_file(path: str = Query(..., description="File path")):
    """
    Download or view a file from an allowed results directory.

    The frontend uses this endpoint both for direct download and for <img src="..."> previews.
    """
    if not is_path_allowed(path):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Path must be within allowed directories: {ALLOWED_BASE_DIRS}",
        )

    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    if file_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is a directory: {path}")

    try:
        return FileResponse(str(file_path), filename=file_path.name)
    except (PermissionError, OSError) as e:
        raise HTTPException(status_code=403, detail=f"Cannot access file: {str(e)}")


@router.get("/search")
async def search_files(
    base_path: Optional[str] = Query(None, description="Base directory to search"),
    pattern: str = Query(..., description="File name pattern (supports wildcards)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results")
):
    """
    Search for files matching a pattern.

    Query Parameters:
        base_path: Base directory to search in
        pattern: Filename pattern (e.g., "*.jpg", "frame_*.png")
        limit: Maximum number of results

    Returns:
        List of matching files

    Example:
        GET /api/results/search?base_path=/mnt/data/extracted/film&pattern=*.jpg&limit=50
    """
    if base_path is None:
        base_path = ALLOWED_BASE_DIRS[0]

    # Security check
    if not is_path_allowed(base_path):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Path must be within allowed directories: {ALLOWED_BASE_DIRS}"
        )

    base_dir = Path(base_path)

    if not base_dir.exists() or not base_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Base directory not found: {base_path}"
        )

    # Search for matching files
    matching_files = []
    count = 0

    try:
        for file_path in base_dir.rglob(pattern):
            if count >= limit:
                break

            if file_path.is_file():
                try:
                    matching_files.append(get_file_info(file_path))
                    count += 1
                except (PermissionError, OSError):
                    continue
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

    return {
        "base_path": base_path,
        "pattern": pattern,
        "total_found": count,
        "limit": limit,
        "files": matching_files
    }


@router.get("/stats")
async def get_directory_stats(
    path: str = Query(..., description="Directory path")
):
    """
    Get statistics for a directory (recursive).

    Query Parameters:
        path: Directory path

    Returns:
        Statistics including total files, total size, file type breakdown

    Example:
        GET /api/results/stats?path=/mnt/data/extracted/film_name
    """
    # Security check
    if not is_path_allowed(path):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Path must be within allowed directories: {ALLOWED_BASE_DIRS}"
        )

    dir_path = Path(path)

    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Directory not found: {path}"
        )

    # Calculate statistics
    total_files = 0
    total_size = 0
    file_types = {}

    try:
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    total_files += 1
                    total_size += stat.st_size

                    # Count by extension
                    ext = file_path.suffix or ".no_extension"
                    file_types[ext] = file_types.get(ext, 0) + 1
                except (PermissionError, OSError):
                    continue
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate stats: {str(e)}"
        )

    return {
        "path": path,
        "total_files": total_files,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "file_types": file_types
    }
