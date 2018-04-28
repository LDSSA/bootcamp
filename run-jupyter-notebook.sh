#!/bin/bash

docker pull ldssa/bootcamp-batch2

docker run  --volume `pwd`:/root/units --workdir /root/units -it --rm -p 127.0.0.1:8888:8888 ldssa/bootcamp-batch2:latest

