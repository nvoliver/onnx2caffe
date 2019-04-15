FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install python3-pip caffe-cpu=1.0.0-6 -y --no-install-recommends
RUN pip3 install setuptools
RUN pip3 install 'argparse==1.4.0' 'numpy==1.13.3' 'protobuf==3.7.1' 'onnx==1.4.0'