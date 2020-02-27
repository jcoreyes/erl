cp /tmp/script_to_scp_over.sh /home/ashvin/code/singularity/scripts/
aws s3 sync --exclude *.git* --exclude *__pycache__* s3://s3doodad/doodad/logs/singularity/scripts/ ./code/singularity/scripts/
