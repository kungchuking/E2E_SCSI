#!/bin/csh

source ./venv/bin/activate.csh

setenv LD_LIBRARY_PATH $HOME/openssl/lib64:$HOME/libffi/lib:$LD_LIBRARY_PATH

