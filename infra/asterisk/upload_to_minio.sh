#!/bin/bash
FILE=$1
EXT=$2
UID=$3
sleep 2
mc alias set myminio http://minio:9000 minioadmin minioadmin123
mc cp "$FILE" myminio/voicegraph-recordings/calls/"$UID".wav
rm -f "$FILE"
