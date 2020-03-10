cp /tmp/script_to_scp_over.sh ./singularity/scripts/
aws s3 sync --exclude *.git* --exclude *__pycache__* ./singularity/scripts/ s3://s3doodad/doodad/logs/singularity/scripts/
