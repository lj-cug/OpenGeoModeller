# Install Firedrake and Thetis

curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install

python3 firedrake-install --install thetis

# The Thetis source will be installed in the src/thetis subdirectory of your Firedrake install. In order to use Firedrake and Thetis you need to activate the Firedrake virtualenv:

source <your-firedrake-install-dir>/firedrake/bin/activate

# If you have already installed Firedrake

You can install Thetis in your Firedrake installation by activating the Firedrake virtualenv and running:

firedrake-update --install thetis

# If you are using a shared, pre-installed Firedrake (such as on some clusters)

Check out the Thetis repository from Github. You then need to add the Thetis repository to your PYTHONPATH in the Firedrake virtualenv. You can do this with pip:

pip install -e <path-to-thetis-repository>
