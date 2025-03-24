#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path
import argparse

def read_version_file(file_path="VERSION"):
    # Get the root directory by going up one level from utils folder
    root_dir = Path(__file__).resolve().parent.parent
    version_path = root_dir / file_path
    
    if not version_path.exists():
        print(f"VERSION file not found at {version_path}")
        sys.exit(1)
    
    with version_path.open() as f:
        version = f.read().strip().strip('"').strip("'")
    
    if not version:
        print("VERSION file is empty or invalid.")
        sys.exit(1)
    
    return version

def run_git_command(args, dry_run=False):
    command = ["git"] + args
    if dry_run:
        print("[DRY RUN] Would run:", " ".join(command))
    else:
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running git command: {' '.join(args)}")
            sys.exit(1)

def check_remote_tag(tag_name):
    try:
        # Get remote tags without fetching them
        remote_refs = subprocess.check_output(
            ["git", "ls-remote", "--tags", "origin", tag_name],
            stderr=subprocess.PIPE
        ).decode().strip()
        return bool(remote_refs)
    except subprocess.CalledProcessError:
        print("Error checking remote tags. Please ensure you have access to the remote repository.")
        sys.exit(1)

def tag_and_push(version, dry_run=False):
    tag_name = f"v{version}"

    # Check if tag exists locally
    existing_tags = subprocess.check_output(["git", "tag"]).decode().splitlines()
    if tag_name in existing_tags:
        print(f"Tag {tag_name} already exists locally.")
        sys.exit(1)

    # Check if tag exists remotely
    if check_remote_tag(tag_name):
        print(f"Tag {tag_name} already exists on remote.")
        sys.exit(1)

    print(f"{'[DRY RUN] ' if dry_run else ''}Creating git tag: {tag_name}")
    run_git_command(["tag", tag_name], dry_run=dry_run)

    print(f"{'[DRY RUN] ' if dry_run else ''}Pushing tag {tag_name} to origin...")
    run_git_command(["push", "origin", tag_name], dry_run=dry_run)

def main():
    parser = argparse.ArgumentParser(description="Tag and push version from VERSION file.")
    parser.add_argument('--dry-run', action='store_true', help='Simulate actions without making changes')
    args = parser.parse_args()

    version = read_version_file()
    tag_and_push(version, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
