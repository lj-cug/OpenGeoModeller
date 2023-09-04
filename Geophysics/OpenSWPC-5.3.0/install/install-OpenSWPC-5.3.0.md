# ¹¹½¨OpenSWPC-5.3.0

https://tktmyd.github.io/OpenSWPC/1._SetUp/0100_trial/

## Python preparation

from IPython.display import HTML
from base64 import b64encode

def show_mp4(mp4file):
  mp4 = open(mp4file, 'rb').read()
  data_url = 'data:video/mp4;base64,' + b64encode(mp4).decode()

  html=HTML(f"""
<video width="50%" height="50%" controls>
   <source src="{data_url}" type="video/mp4">
</video>""")

  return html

  
## Download & Compile OpenSWPC

apt install libnetcdf-dev libnetcdff-dev

git clone https://github.com/tktmyd/OpenSWPC.git

cd ./OpenSWPC/src; make arch=ubuntu-gfortran

After long messages, we get executable binaries under ./OpenSWPC/bin/ directory:

ls ./OpenSWPC/bin

# Execution

mpirun --allow-run-as-root -np 2 ./bin/swpc_psv.x -i example/input.inf

The warning messages regarding floating-point underflow do not matter.

# Visualization

OpenSWPC also comes with dedicated tools for visualization.

./bin/read_snp.x -i out/swpc.xz.v.nc -ppm -pall -mul 10
 
Let us confirm the result as an animation movie by using ffmpeg.

ffmpeg -i ppm/swpc.xz.v2.%6d.ppm -qscale 0 -pix_fmt yuv420p -y swpc.mp4 

show_mp4("./swpc.mp4")