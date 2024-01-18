#!/bin/bash
eval "$(conda shell.bash hook)"

cd checkpoint
curl -O -L https://huggingface.co/xingxm/PyTorch-SVGRender-models/resolve/main/u2net.zip
unzip u2net.zip

echo "U^2Net download success"
