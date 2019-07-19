#!/bin/bash 
echo "evaluate faces statistics"

size=160
margin=32
detector=frcnnv3

python -m facenet.apps.faces_statistics \
 ~/datasets/vggface2/train/ \
 --detector ${detector} \
