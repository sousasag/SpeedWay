SHELL = /bin/bash

all:
	cd tmcalc_cython; python setup.py build_ext --inplace

clean:
	cd tmcalc_cython; rm -rf build tmcalc_module.so tmcalc_module.c

