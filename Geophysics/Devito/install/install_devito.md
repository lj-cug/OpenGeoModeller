# Create new env with the name devito
conda create --name devito

# Activate the environment
conda activate devito

```
pip install devito
# ... or to install additional dependencies
# pip install devito[extras,mpi,nvidia,tests]
```
# Install from source
```
git clone https://github.com/devitocodes/devito.git
cd devito

# Install requirements
pip install -e .

# ...or to install additional dependencies
# pip install -e .[extras,mpi,nvidia,tests]
```
