# Install stride

git clone https://github.com/trustimaging/stride.git

cd stride

conda env create -f environment.yml

conda activate stride

pip install -e .

## ʹ��Docker����

git clone https://github.com/trustimaging/stride.git

cd stride

docker-compose up stride

