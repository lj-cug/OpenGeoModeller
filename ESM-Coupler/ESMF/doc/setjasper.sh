ungrid_DIR=/home/lijian/ESM_lj

export JASPERINC=${ungrid_DIR}/jasper/include/
export JASPERINC="${JASPERINC} -I${ungrid_DIR}/libpng16/include/"
export JASPERINC="${JASPERINC} -I${ungrid_DIR}/zlib/include/"
export JASPERLIB=${ungrid_DIR}/jasper/lib/
export JASPERLIB="${JASPERLIB} -L${ungrid_DIR}/libpng16/lib/"
export JASPERLIB="${JASPERLIB} -L${ungrid_DIR}/zlib/lib/"
