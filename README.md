# Convert ONNX to Caffe
This tool converts [ONNX](https://github.com/onnx/onnx) models to [Caffe](https://github.com/BVLC/caffe) format. Only use for inference.

## Dependencies
* Caffe 1.0 (with Python support)
* ONNX 1.4.0
* protobuf 3.7.1  

## How to use
To convert an ONNX model to Caffe:
```
python convertCaffe.py --onnx ./model/MobileNetV2.onnx --prototxt ./model/MobileNetV2.prototxt --caffemodel ./model/MobileNetV2.caffemodel
```

### Setup with Docker
Build a Docker image with a minimal Caffe 1.0, ONNX 1.4.0 and protobuf 3.7.1  installation, using the `Dockerfile` inside this repo:
```
docker build --network=host -t onnx-caffe_18.04 .
```

Then create a Docker container based on this image and open a terminal inside:
```
docker run --network=host -e PYTHONPATH=/workspace/onnx2caffe -v $PWD:/workspace -it onnx-caffe_18.04 /bin/bash
cd /workspace
python3 convertCaffe.py [with arguments from above]
```

### Currently supported operations
* Conv
* ConvTranspose
* BatchNormalization
* MaxPool
* AveragePool
* Leaky ReLU
* ReLU
* Sigmoid
* Dropout
* Gemm (InnerProduct only)
* Add
* Mul
* Reshape
* Upsample
* Concat
* Flatten
