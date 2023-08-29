git clone https://github.com/SeisSol/PSpaMM.git
# make sure $HOME/bin exists or create it with "mkdir ~/bin"
ln -s $(pwd)/PSpaMM/pspamm.py $HOME/bin/pspamm.py

# Instead of linking, you could also add the following line to your .bashrc:
export PATH=<Your_Path_to_PSpaMM>:$PATH
