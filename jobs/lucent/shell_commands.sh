# Preferably execute commands one by one for more controllability

# Conda Environment
conda create --prefix /scratch/vihps/vihps01/vit-mlp-explainability/env python=3.9
conda activate /scratch/vihps/vihps01/vit-mlp-explainability/env

# Get Source Code and mapping files to server
cd /scratch/vihps/vihps01/vit-mlp-explainability/
git clone https://github.com/SoulofAkuma/dlcv-vit-mlp-explainability ./code
cd code
git checkout cluster
python -m pip install -r requirements.txt
python -m pip install .

# Execute SLURM Job for generating the configs
cd /scratch/vihps/vihps01/vit-mlp-explainability/
sbatch ./code/jobs/lucent/jobscript_configs.sh

# Execute SLURM Job for generating the images
cd /scratch/vihps/vihps01/vit-mlp-explainability/
sbatch ./code/jobs/lucent/jobscript_images.sh

# Generate the job configs (Preferably do this via a slurm job aswell because this loads a model)
# cd /scratch/vihps/vihps01/vit-mlp-explainability/
# python ./code/jobs/lucent/generate_configs.py --output-dir /scratch/vihps/vihps01/vit-mlp-explainability/configs

# Get Imagenet Data to Server (currently not necessary)
# cd /scratch/vihps/vihps01
# mkdir -p imagenet-1k
# scp -r ./path/to/local/imagenet/dataset/val /scratch/vihps/vihps01/imagenet-1k