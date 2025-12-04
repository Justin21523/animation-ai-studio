#!/usr/bin/env python3
"""
File Organizer (檔案組織器)
===========================

Provides comprehensive file organization and management capabilities.
All operations are CPU-only and optimized for large-scale file operations.

Features:
- Organize files by type, date, size
- Batch rename with pattern matching
- Duplicate file detection (content hash)
- Directory synchronization
- Advanced file search
- Disk space analysis

Author: Animation AI Studio Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import shutil
import hashlib
import re
import argparse
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import mimetypes
import fnmatch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import safety infrastructure from Phase 1
try:
    from scripts.core.safety import MemoryMonitor, run_preflight
except ImportError:
    print("⚠️ Warning: Safety infrastructure not found, using stub implementation")
    class MemoryMonitor:
        def __init__(self, **kwargs):
            pass
        def check_memory(self):
            return 'ok', 0.0
    def run_preflight():
        return True

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FileInfo:
    """File information (檔案資訊)"""
    path: str
    name: str
    size_bytes: int
    modified_time: float
    created_time: float
    extension: str
    mime_type: Optional[str] = None
    hash_md5: Optional[str] = None
    hash_sha256: Optional[str] = None

@dataclass
class OrganizeResult:
    """Organization result (組織結果)"""
    success: bool
    files_processed: int
    files_moved: int
    files_copied: int
    files_skipped: int
    errors: List[str]
    summary: Dict[str, int]

@dataclass
class DuplicateGroup:
    """Duplicate file group (重複檔案群組)"""
    hash: str
    size_bytes: int
    files: List[str]
    total_wasted_space: int

# ============================================================================
# File Organizer Class
# ============================================================================

class FileOrganizer:
    """
    File Organizer (檔案組織器)

    Provides comprehensive file organization and management capabilities.
    """

    # File type categories
    FILE_CATEGORIES = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'],
        'videos': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'],
        'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'],
        'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.pages'],
        'spreadsheets': ['.xls', '.xlsx', '.csv', '.ods', '.numbers'],
        'presentations': ['.ppt', '.pptx', '.key', '.odp'],
        'archives': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h', '.sh', '.yaml', '.json'],
        'executables': ['.exe', '.app', '.dmg', '.deb', '.rpm', '.apk'],
    }

    def __init__(self, dry_run: bool = False):
        """
        Initialize File Organizer.
        初始化檔案組織器。

        Args:
            dry_run: If True, only simulate operations without actual changes
                    如果為 True，僅模擬操作而不實際變更
        """
        self.dry_run = dry_run
        self.memory_monitor = MemoryMonitor()

        # Initialize mimetypes
        mimetypes.init()

    # ========================================================================
    # Core Operations
    # ========================================================================

    def organize_by_type(
        self,
        input_dir: str,
        output_dir: str,
        create_subdirs: bool = True,
        move_files: bool = False
    ) -> OrganizeResult:
        """
        Organize files by type into category folders.
        按類型將檔案組織到分類資料夾。

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            create_subdirs: Create subdirectories for each category
            move_files: Move files instead of copy

        Returns:
            OrganizeResult with statistics
        """
        print(f"📂 {'Moving' if move_files else 'Copying'} files from {input_dir} to {output_dir}")
        print(f"   Organizing by type {'with' if create_subdirs else 'without'} subdirectories")

        if self.dry_run:
            print("🔍 DRY RUN MODE - No actual changes will be made")

        # Check memory (basic check)
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                print(f"⚠️ Warning: Low memory ({mem.percent:.1f}% used)")
        except:
            pass  # Skip memory check if psutil not available

        # Initialize result
        result = OrganizeResult(
            success=True,
            files_processed=0,
            files_moved=0 if not move_files else 0,
            files_copied=0 if move_files else 0,
            files_skipped=0,
            errors=[],
            summary=defaultdict(int)
        )

        # Create output directory
        if not self.dry_run:
            os.makedirs(output_dir, exist_ok=True)

        try:
            # Get all files
            files = self._get_all_files(input_dir)
            print(f"📊 Found {len(files)} files to organize")

            # Process each file
            for file_path in files:
                result.files_processed += 1

                try:
                    # Get file info
                    file_info = self._get_file_info(file_path)

                    # Determine category
                    category = self._get_file_category(file_info.extension)
                    result.summary[category] += 1

                    # Determine target directory
                    if create_subdirs:
                        target_dir = os.path.join(output_dir, category)
                    else:
                        target_dir = output_dir

                    # Create target directory
                    if not self.dry_run:
                        os.makedirs(target_dir, exist_ok=True)

                    # Determine target path
                    target_path = os.path.join(target_dir, file_info.name)

                    # Handle name conflicts
                    target_path = self._handle_name_conflict(target_path)

                    # Move or copy
                    if self.dry_run:
                        print(f"   [DRY RUN] {'Move' if move_files else 'Copy'}: {file_path} → {target_path}")
                    else:
                        if move_files:
                            shutil.move(file_path, target_path)
                            result.files_moved += 1
                        else:
                            shutil.copy2(file_path, target_path)
                            result.files_copied += 1

                except Exception as e:
                    result.errors.append(f"Error processing {file_path}: {str(e)}")
                    result.files_skipped += 1

            # Print summary
            print("\n✅ Organization complete!")
            print(f"   Files processed: {result.files_processed}")
            print(f"   Files {'moved' if move_files else 'copied'}: {result.files_moved if move_files else result.files_copied}")
            print(f"   Files skipped: {result.files_skipped}")

            if result.summary:
                print("\n📊 Category breakdown:")
                for category, count in sorted(result.summary.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {category}: {count} files")

        except Exception as e:
            result.success = False
            result.errors.append(f"Fatal error: {str(e)}")
            print(f"❌ Error: {str(e)}")

        return result

    def organize_by_date(
        self,
        input_dir: str,
        output_dir: str,
        date_format: str = "%Y/%m",
        use_modified_date: bool = True,
        move_files: bool = False
    ) -> OrganizeResult:
        """
        Organize files by date into YYYY/MM folders.
        按日期將檔案組織到 YYYY/MM 資料夾。

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            date_format: Date format for folder structure (e.g., "%Y/%m/%d")
            use_modified_date: Use modified date (True) or created date (False)
            move_files: Move files instead of copy

        Returns:
            OrganizeResult with statistics
        """
        print(f"📅 Organizing files by {'modified' if use_modified_date else 'created'} date")
        print(f"   Date format: {date_format}")

        if self.dry_run:
            print("🔍 DRY RUN MODE - No actual changes will be made")

        # Initialize result
        result = OrganizeResult(
            success=True,
            files_processed=0,
            files_moved=0 if not move_files else 0,
            files_copied=0 if move_files else 0,
            files_skipped=0,
            errors=[],
            summary=defaultdict(int)
        )

        # Create output directory
        if not self.dry_run:
            os.makedirs(output_dir, exist_ok=True)

        try:
            # Get all files
            files = self._get_all_files(input_dir)
            print(f"📊 Found {len(files)} files to organize")

            # Process each file
            for file_path in files:
                result.files_processed += 1

                try:
                    # Get file info
                    file_info = self._get_file_info(file_path)

                    # Determine date
                    timestamp = file_info.modified_time if use_modified_date else file_info.created_time
                    date_str = datetime.fromtimestamp(timestamp).strftime(date_format)
                    result.summary[date_str] += 1

                    # Determine target directory
                    target_dir = os.path.join(output_dir, date_str)

                    # Create target directory
                    if not self.dry_run:
                        os.makedirs(target_dir, exist_ok=True)

                    # Determine target path
                    target_path = os.path.join(target_dir, file_info.name)
                    target_path = self._handle_name_conflict(target_path)

                    # Move or copy
                    if self.dry_run:
                        print(f"   [DRY RUN] {'Move' if move_files else 'Copy'}: {file_path} → {target_path}")
                    else:
                        if move_files:
                            shutil.move(file_path, target_path)
                            result.files_moved += 1
                        else:
                            shutil.copy2(file_path, target_path)
                            result.files_copied += 1

                except Exception as e:
                    result.errors.append(f"Error processing {file_path}: {str(e)}")
                    result.files_skipped += 1

            # Print summary
            print("\n✅ Organization complete!")
            print(f"   Files processed: {result.files_processed}")
            print(f"   Files {'moved' if move_files else 'copied'}: {result.files_moved if move_files else result.files_copied}")

            if result.summary:
                print("\n📊 Date breakdown:")
                for date_str, count in sorted(result.summary.items()):
                    print(f"   {date_str}: {count} files")

        except Exception as e:
            result.success = False
            result.errors.append(f"Fatal error: {str(e)}")
            print(f"❌ Error: {str(e)}")

        return result

    def batch_rename(
        self,
        input_dir: str,
        pattern: str,
        replacement: str,
        use_regex: bool = False,
        recursive: bool = False
    ) -> Dict[str, str]:
        """
        Batch rename files using pattern matching.
        使用模式匹配批次重命名檔案。

        Args:
            input_dir: Input directory path
            pattern: Pattern to match (glob or regex)
            replacement: Replacement pattern
            use_regex: Use regex instead of glob
            recursive: Process subdirectories recursively

        Returns:
            Dictionary mapping old paths to new paths
        """
        print(f"✏️ Batch renaming files in {input_dir}")
        print(f"   Pattern: {pattern} → {replacement}")
        print(f"   Using {'regex' if use_regex else 'glob'} matching")

        if self.dry_run:
            print("🔍 DRY RUN MODE - No actual changes will be made")

        renamed_files = {}

        try:
            # Get all files
            if recursive:
                files = self._get_all_files(input_dir)
            else:
                files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                        if os.path.isfile(os.path.join(input_dir, f))]

            print(f"📊 Found {len(files)} files to process")

            # Process each file
            for file_path in files:
                try:
                    file_name = os.path.basename(file_path)
                    dir_name = os.path.dirname(file_path)

                    # Apply pattern matching
                    if use_regex:
                        new_name = re.sub(pattern, replacement, file_name)
                    else:
                        if fnmatch.fnmatch(file_name, pattern):
                            new_name = replacement
                        else:
                            continue

                    # Skip if name didn't change
                    if new_name == file_name:
                        continue

                    # Determine new path
                    new_path = os.path.join(dir_name, new_name)
                    new_path = self._handle_name_conflict(new_path)

                    # Rename
                    if self.dry_run:
                        print(f"   [DRY RUN] Rename: {file_name} → {os.path.basename(new_path)}")
                    else:
                        os.rename(file_path, new_path)
                        print(f"   ✅ Renamed: {file_name} → {os.path.basename(new_path)}")

                    renamed_files[file_path] = new_path

                except Exception as e:
                    print(f"   ❌ Error renaming {file_path}: {str(e)}")

            print(f"\n✅ Renamed {len(renamed_files)} files")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

        return renamed_files

    def find_duplicates(
        self,
        input_dir: str,
        method: str = 'hash',
        recursive: bool = True,
        min_size: int = 0
    ) -> List[DuplicateGroup]:
        """
        Find duplicate files by content hash or name.
        按內容雜湊或名稱尋找重複檔案。

        Args:
            input_dir: Input directory path
            method: Detection method ('hash', 'name', 'size')
            recursive: Process subdirectories recursively
            min_size: Minimum file size to consider (bytes)

        Returns:
            List of duplicate groups
        """
        print(f"🔍 Finding duplicates in {input_dir}")
        print(f"   Method: {method}")
        print(f"   Minimum size: {self._format_size(min_size)}")

        duplicates = []

        try:
            # Get all files
            if recursive:
                files = self._get_all_files(input_dir)
            else:
                files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                        if os.path.isfile(os.path.join(input_dir, f))]

            # Filter by size
            files = [f for f in files if os.path.getsize(f) >= min_size]

            print(f"📊 Analyzing {len(files)} files...")

            # Group files by detection method
            groups = defaultdict(list)

            for file_path in files:
                try:
                    if method == 'hash':
                        # Use MD5 hash for duplicate detection
                        file_hash = self._calculate_md5(file_path)
                        groups[file_hash].append(file_path)
                    elif method == 'name':
                        # Group by filename
                        file_name = os.path.basename(file_path)
                        groups[file_name].append(file_path)
                    elif method == 'size':
                        # Group by file size
                        file_size = os.path.getsize(file_path)
                        groups[file_size].append(file_path)

                except Exception as e:
                    print(f"   ⚠️ Warning: Error processing {file_path}: {str(e)}")

            # Find duplicates (groups with more than one file)
            for key, file_list in groups.items():
                if len(file_list) > 1:
                    size_bytes = os.path.getsize(file_list[0])
                    total_wasted = size_bytes * (len(file_list) - 1)

                    duplicates.append(DuplicateGroup(
                        hash=str(key),
                        size_bytes=size_bytes,
                        files=file_list,
                        total_wasted_space=total_wasted
                    ))

            # Sort by wasted space
            duplicates.sort(key=lambda x: x.total_wasted_space, reverse=True)

            # Print results
            if duplicates:
                print(f"\n🔍 Found {len(duplicates)} duplicate groups")
                total_wasted = sum(d.total_wasted_space for d in duplicates)
                print(f"   Total wasted space: {self._format_size(total_wasted)}")

                print("\n📊 Top duplicate groups:")
                for i, dup in enumerate(duplicates[:10], 1):
                    print(f"\n   {i}. Group (wasted: {self._format_size(dup.total_wasted_space)})")
                    for file_path in dup.files[:5]:
                        print(f"      - {file_path}")
                    if len(dup.files) > 5:
                        print(f"      ... and {len(dup.files) - 5} more")
            else:
                print("✅ No duplicates found!")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

        return duplicates

    def analyze_disk_space(
        self,
        input_dir: str,
        depth: int = 2,
        top_n: int = 20
    ) -> Dict:
        """
        Analyze disk space usage by directory and file type.
        按目錄和檔案類型分析磁碟空間使用。

        Args:
            input_dir: Input directory path
            depth: Maximum directory depth to analyze
            top_n: Number of top items to show

        Returns:
            Dictionary with analysis results
        """
        print(f"📊 Analyzing disk space in {input_dir}")
        print(f"   Depth: {depth}, Top items: {top_n}")

        analysis = {
            'total_size': 0,
            'total_files': 0,
            'total_dirs': 0,
            'by_type': defaultdict(lambda: {'count': 0, 'size': 0}),
            'by_dir': {},
            'largest_files': [],
        }

        try:
            # Collect all files and directories
            all_files = []

            for root, dirs, files in os.walk(input_dir):
                # Calculate depth
                current_depth = root[len(input_dir):].count(os.sep)
                if current_depth > depth:
                    continue

                analysis['total_dirs'] += len(dirs)

                # Process files
                for file_name in files:
                    file_path = os.path.join(root, file_name)

                    try:
                        file_size = os.path.getsize(file_path)
                        analysis['total_size'] += file_size
                        analysis['total_files'] += 1

                        # Track by type
                        ext = os.path.splitext(file_name)[1].lower()
                        analysis['by_type'][ext]['count'] += 1
                        analysis['by_type'][ext]['size'] += file_size

                        # Track largest files
                        all_files.append((file_path, file_size))

                    except Exception as e:
                        print(f"   ⚠️ Warning: Error processing {file_path}: {str(e)}")

            # Calculate directory sizes
            for root, dirs, files in os.walk(input_dir):
                current_depth = root[len(input_dir):].count(os.sep)
                if current_depth > depth:
                    continue

                dir_size = sum(os.path.getsize(os.path.join(root, f)) for f in files
                             if os.path.isfile(os.path.join(root, f)))
                analysis['by_dir'][root] = dir_size

            # Sort largest files
            all_files.sort(key=lambda x: x[1], reverse=True)
            analysis['largest_files'] = all_files[:top_n]

            # Print results
            print(f"\n📊 Analysis Results")
            print(f"   Total size: {self._format_size(analysis['total_size'])}")
            print(f"   Total files: {analysis['total_files']:,}")
            print(f"   Total directories: {analysis['total_dirs']:,}")

            # Print by type
            print(f"\n📂 Top {top_n} file types by size:")
            sorted_types = sorted(analysis['by_type'].items(),
                                key=lambda x: x[1]['size'], reverse=True)
            for i, (ext, data) in enumerate(sorted_types[:top_n], 1):
                ext_name = ext if ext else '(no extension)'
                print(f"   {i}. {ext_name}: {data['count']:,} files, {self._format_size(data['size'])}")

            # Print largest directories
            print(f"\n📁 Top {top_n} largest directories:")
            sorted_dirs = sorted(analysis['by_dir'].items(),
                               key=lambda x: x[1], reverse=True)
            for i, (dir_path, size) in enumerate(sorted_dirs[:top_n], 1):
                rel_path = os.path.relpath(dir_path, input_dir)
                print(f"   {i}. {rel_path}: {self._format_size(size)}")

            # Print largest files
            print(f"\n📄 Top {top_n} largest files:")
            for i, (file_path, size) in enumerate(analysis['largest_files'], 1):
                rel_path = os.path.relpath(file_path, input_dir)
                print(f"   {i}. {rel_path}: {self._format_size(size)}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

        return analysis

    def search_files(
        self,
        input_dir: str,
        name_pattern: Optional[str] = None,
        extension: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        modified_after: Optional[str] = None,
        modified_before: Optional[str] = None,
        recursive: bool = True
    ) -> List[str]:
        """
        Search files with advanced filters.
        使用進階篩選搜尋檔案。

        Args:
            input_dir: Input directory path
            name_pattern: Filename pattern (glob)
            extension: File extension filter
            min_size: Minimum file size (bytes)
            max_size: Maximum file size (bytes)
            modified_after: Modified after date (YYYY-MM-DD)
            modified_before: Modified before date (YYYY-MM-DD)
            recursive: Process subdirectories recursively

        Returns:
            List of matching file paths
        """
        print(f"🔍 Searching files in {input_dir}")

        filters = []
        if name_pattern:
            filters.append(f"name: {name_pattern}")
        if extension:
            filters.append(f"ext: {extension}")
        if min_size:
            filters.append(f"min: {self._format_size(min_size)}")
        if max_size:
            filters.append(f"max: {self._format_size(max_size)}")
        if modified_after:
            filters.append(f"after: {modified_after}")
        if modified_before:
            filters.append(f"before: {modified_before}")

        print(f"   Filters: {', '.join(filters) if filters else 'none'}")

        matches = []

        try:
            # Get all files
            if recursive:
                files = self._get_all_files(input_dir)
            else:
                files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                        if os.path.isfile(os.path.join(input_dir, f))]

            # Convert date filters
            after_timestamp = None
            before_timestamp = None

            if modified_after:
                after_timestamp = datetime.strptime(modified_after, "%Y-%m-%d").timestamp()
            if modified_before:
                before_timestamp = datetime.strptime(modified_before, "%Y-%m-%d").timestamp()

            # Apply filters
            for file_path in files:
                try:
                    # Name pattern filter
                    if name_pattern:
                        if not fnmatch.fnmatch(os.path.basename(file_path), name_pattern):
                            continue

                    # Extension filter
                    if extension:
                        if not file_path.lower().endswith(extension.lower()):
                            continue

                    # Size filters
                    file_size = os.path.getsize(file_path)
                    if min_size and file_size < min_size:
                        continue
                    if max_size and file_size > max_size:
                        continue

                    # Date filters
                    file_mtime = os.path.getmtime(file_path)
                    if after_timestamp and file_mtime < after_timestamp:
                        continue
                    if before_timestamp and file_mtime > before_timestamp:
                        continue

                    # All filters passed
                    matches.append(file_path)

                except Exception as e:
                    print(f"   ⚠️ Warning: Error processing {file_path}: {str(e)}")

            # Print results
            print(f"\n✅ Found {len(matches)} matching files")

            if matches:
                print("\n📄 Sample matches:")
                for file_path in matches[:20]:
                    rel_path = os.path.relpath(file_path, input_dir)
                    size = os.path.getsize(file_path)
                    print(f"   - {rel_path} ({self._format_size(size)})")

                if len(matches) > 20:
                    print(f"   ... and {len(matches) - 20} more")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

        return matches

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_all_files(self, directory: str) -> List[str]:
        """
        Get all files in directory recursively.
        遞迴取得目錄中的所有檔案。
        """
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return files

    def _get_file_info(self, file_path: str) -> FileInfo:
        """
        Get file information.
        取得檔案資訊。
        """
        stat = os.stat(file_path)
        name = os.path.basename(file_path)
        ext = os.path.splitext(name)[1].lower()
        mime_type, _ = mimetypes.guess_type(file_path)

        return FileInfo(
            path=file_path,
            name=name,
            size_bytes=stat.st_size,
            modified_time=stat.st_mtime,
            created_time=stat.st_ctime,
            extension=ext,
            mime_type=mime_type
        )

    def _get_file_category(self, extension: str) -> str:
        """
        Get file category based on extension.
        根據副檔名取得檔案分類。
        """
        for category, extensions in self.FILE_CATEGORIES.items():
            if extension.lower() in extensions:
                return category
        return 'other'

    def _handle_name_conflict(self, file_path: str) -> str:
        """
        Handle filename conflicts by appending number.
        透過附加數字處理檔名衝突。
        """
        if not os.path.exists(file_path):
            return file_path

        base, ext = os.path.splitext(file_path)
        counter = 1

        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1

        return f"{base}_{counter}{ext}"

    def _calculate_md5(self, file_path: str, chunk_size: int = 8192) -> str:
        """
        Calculate MD5 hash of file.
        計算檔案的 MD5 雜湊。
        """
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()

    def _calculate_sha256(self, file_path: str, chunk_size: int = 8192) -> str:
        """
        Calculate SHA256 hash of file.
        計算檔案的 SHA256 雜湊。
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _format_size(self, size_bytes: int) -> str:
        """
        Format file size in human-readable format.
        格式化檔案大小為人類可讀格式。
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

# ============================================================================
# CLI Interface
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser. 建立參數解析器。"""
    parser = argparse.ArgumentParser(
        description='File Organizer - Comprehensive file organization and management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (範例):

  # Organize files by type
  python file_organizer.py organize-by-type --input /path/to/files --output /path/to/organized

  # Organize files by date
  python file_organizer.py organize-by-date --input /path/to/files --output /path/to/by_date

  # Batch rename files
  python file_organizer.py batch-rename --input /path/to/files --pattern "*.txt" --replacement "doc_{}.txt" --use-regex

  # Find duplicate files
  python file_organizer.py find-duplicates --input /path/to/files --method hash

  # Analyze disk space
  python file_organizer.py analyze-disk-space --input /path/to/analyze --depth 3 --top-n 20

  # Search files
  python file_organizer.py search --input /path/to/search --name-pattern "*.jpg" --min-size 1048576
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Common arguments
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate operations without making changes')
    parser.add_argument('--skip-preflight', action='store_true',
                       help='Skip preflight checks')

    # Organize by type
    organize_type = subparsers.add_parser('organize-by-type', help='Organize files by type')
    organize_type.add_argument('--input', required=True, help='Input directory')
    organize_type.add_argument('--output', required=True, help='Output directory')
    organize_type.add_argument('--no-subdirs', action='store_true',
                              help='Do not create subdirectories for categories')
    organize_type.add_argument('--move', action='store_true',
                              help='Move files instead of copy')

    # Organize by date
    organize_date = subparsers.add_parser('organize-by-date', help='Organize files by date')
    organize_date.add_argument('--input', required=True, help='Input directory')
    organize_date.add_argument('--output', required=True, help='Output directory')
    organize_date.add_argument('--date-format', default='%Y/%m',
                              help='Date format (default: %%Y/%%m)')
    organize_date.add_argument('--use-created-date', action='store_true',
                              help='Use created date instead of modified date')
    organize_date.add_argument('--move', action='store_true',
                              help='Move files instead of copy')

    # Batch rename
    rename = subparsers.add_parser('batch-rename', help='Batch rename files')
    rename.add_argument('--input', required=True, help='Input directory')
    rename.add_argument('--pattern', required=True, help='Pattern to match')
    rename.add_argument('--replacement', required=True, help='Replacement pattern')
    rename.add_argument('--use-regex', action='store_true',
                       help='Use regex instead of glob')
    rename.add_argument('--recursive', action='store_true',
                       help='Process subdirectories recursively')

    # Find duplicates
    duplicates = subparsers.add_parser('find-duplicates', help='Find duplicate files')
    duplicates.add_argument('--input', required=True, help='Input directory')
    duplicates.add_argument('--method', default='hash',
                           choices=['hash', 'name', 'size'],
                           help='Detection method')
    duplicates.add_argument('--no-recursive', action='store_true',
                           help='Do not process subdirectories')
    duplicates.add_argument('--min-size', type=int, default=0,
                           help='Minimum file size (bytes)')
    duplicates.add_argument('--output-json', help='Output JSON file path')

    # Analyze disk space
    analyze = subparsers.add_parser('analyze-disk-space', help='Analyze disk space usage')
    analyze.add_argument('--input', required=True, help='Input directory')
    analyze.add_argument('--depth', type=int, default=2,
                        help='Maximum directory depth')
    analyze.add_argument('--top-n', type=int, default=20,
                        help='Number of top items to show')
    analyze.add_argument('--output-json', help='Output JSON file path')

    # Search files
    search = subparsers.add_parser('search', help='Search files with filters')
    search.add_argument('--input', required=True, help='Input directory')
    search.add_argument('--name-pattern', help='Filename pattern (glob)')
    search.add_argument('--extension', help='File extension filter')
    search.add_argument('--min-size', type=int, help='Minimum file size (bytes)')
    search.add_argument('--max-size', type=int, help='Maximum file size (bytes)')
    search.add_argument('--modified-after', help='Modified after (YYYY-MM-DD)')
    search.add_argument('--modified-before', help='Modified before (YYYY-MM-DD)')
    search.add_argument('--no-recursive', action='store_true',
                       help='Do not process subdirectories')
    search.add_argument('--output-list', help='Output file list to text file')

    return parser

def main():
    """Main entry point. 主要進入點。"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Run preflight checks
    if not args.skip_preflight:
        if not run_preflight():
            print("❌ Preflight checks failed!")
            return 1

    # Initialize organizer
    organizer = FileOrganizer(dry_run=args.dry_run)

    # Execute command
    try:
        if args.command == 'organize-by-type':
            result = organizer.organize_by_type(
                input_dir=args.input,
                output_dir=args.output,
                create_subdirs=not args.no_subdirs,
                move_files=args.move
            )
            return 0 if result.success else 1

        elif args.command == 'organize-by-date':
            result = organizer.organize_by_date(
                input_dir=args.input,
                output_dir=args.output,
                date_format=args.date_format,
                use_modified_date=not args.use_created_date,
                move_files=args.move
            )
            return 0 if result.success else 1

        elif args.command == 'batch-rename':
            renamed_files = organizer.batch_rename(
                input_dir=args.input,
                pattern=args.pattern,
                replacement=args.replacement,
                use_regex=args.use_regex,
                recursive=args.recursive
            )
            return 0

        elif args.command == 'find-duplicates':
            duplicates = organizer.find_duplicates(
                input_dir=args.input,
                method=args.method,
                recursive=not args.no_recursive,
                min_size=args.min_size
            )

            if args.output_json:
                with open(args.output_json, 'w') as f:
                    json.dump([asdict(d) for d in duplicates], f, indent=2)
                print(f"✅ Saved results to {args.output_json}")

            return 0

        elif args.command == 'analyze-disk-space':
            analysis = organizer.analyze_disk_space(
                input_dir=args.input,
                depth=args.depth,
                top_n=args.top_n
            )

            if args.output_json:
                # Convert defaultdict to regular dict for JSON serialization
                analysis['by_type'] = dict(analysis['by_type'])
                with open(args.output_json, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
                print(f"✅ Saved results to {args.output_json}")

            return 0

        elif args.command == 'search':
            matches = organizer.search_files(
                input_dir=args.input,
                name_pattern=args.name_pattern,
                extension=args.extension,
                min_size=args.min_size,
                max_size=args.max_size,
                modified_after=args.modified_after,
                modified_before=args.modified_before,
                recursive=not args.no_recursive
            )

            if args.output_list:
                with open(args.output_list, 'w') as f:
                    f.write('\n'.join(matches))
                print(f"✅ Saved file list to {args.output_list}")

            return 0

    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
