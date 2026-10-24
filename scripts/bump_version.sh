#!/bin/bash

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 <new_version>"
    exit 1
fi

NEW_VERSION=$1
if [[ ! "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Invalid version '$NEW_VERSION'; expected X.Y.Z"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

sed_edit() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -E -i '' "$@"
    else
        sed -E -i "$@"
    fi
}

echo "Updating versions in $PROJECT_ROOT/packages/"
while IFS= read -r -d '' file; do
    sed_edit "s/^version = \"[^\"]+\"/version = \"${NEW_VERSION}\"/" "$file"
    sed_edit "s/(\"(leann-core|leann-backend-hnsw|leann-backend-diskann|leann-backend-ivf)(\[[^]]+\])?[=<>~!]*)[0-9]+\.[0-9]+\.[0-9]+/\1${NEW_VERSION}/g" "$file"
done < <(find "$PROJECT_ROOT/packages" -name "pyproject.toml" -print0)

for file in "$PROJECT_ROOT/docker/Dockerfile" "$PROJECT_ROOT/docker/Dockerfile.cpu"; do
    sed_edit "s/^ARG LEANN_VERSION=.*/ARG LEANN_VERSION=${NEW_VERSION}/" "$file"
done
sed_edit "s/(--build-arg LEANN_VERSION=)[0-9]+\.[0-9]+\.[0-9]+/\1${NEW_VERSION}/g" \
    "$PROJECT_ROOT/docker/README.md"

echo "✅ Package versions and internal dependencies updated to $NEW_VERSION"
echo "✅ Docker defaults and examples updated to $NEW_VERSION"
