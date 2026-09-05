#!/bin/bash
if [ -z "$BASE_LIBS_PATH" ]; then
    if [ -z "$ASCEND_HOME_PATH" ]; then
        if [ -z "$ASCEND_AICPU_PATH" ]; then
            # 兜底：与CMakePresets.json中ASCEND_CANN_PACKAGE_PATH保持一致的CANN 9.0.0安装路径
            if [ -d "/usr/local/Ascend/cann-9.0.0" ]; then
                export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
            elif [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
                export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
            elif [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
                export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
            else
                echo "please set env."
                exit 1
            fi
        else
            export ASCEND_HOME_PATH=$ASCEND_AICPU_PATH
        fi
    else
        export ASCEND_HOME_PATH=$ASCEND_HOME_PATH
    fi
else
    export ASCEND_HOME_PATH=$BASE_LIBS_PATH
fi
echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
script_path=$(realpath $(dirname $0))

BUILD_DIR="build_out"
HOST_NATIVE_DIR="host_native_tiling"
mkdir -p build_out
rm -rf build_out/*

ENABLE_CROSS="-DENABLE_CROSS_COMPILE=True"
ENABLE_BINARY="-DENABLE_BINARY_PACKAGE=True"
ENABLE_LIBRARY="-DASCEND_PACK_SHARED_LIBRARY=True"
cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')

target=package
if [ "$1"x != ""x ]; then target=$1; fi

cmake -S . -B "$BUILD_DIR" --preset=default
cmake --build "$BUILD_DIR" --target binary -j$(nproc)
cmake --build "$BUILD_DIR" --target $target -j$(nproc)