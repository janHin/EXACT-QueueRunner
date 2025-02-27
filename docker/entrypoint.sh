#!/bin/sh

# Show if we have GPU support in the container
/usr/bin/nvidia-smi

exec "$@"
