#!/bin/bash
set -e

# Wait for any initialization if needed
echo "Starting Runner Air Planner..."

# Execute the command passed to the container
exec "$@"

