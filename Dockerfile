FROM nvidia/cuda:9.2-base-ubuntu18.04
LABEL maintainer="David Uhm <david.uhm@lge.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        vim \
        gnupg2 \
        lsb-release \
        build-essential \
        cuda-command-line-tools-9-2 \
        cuda-cublas-dev-9-2 \
        cuda-cudart-dev-9-2 \
        cuda-cufft-dev-9-2 \
        cuda-curand-dev-9-2 \
        cuda-cusolver-dev-9-2 \
        cuda-cusparse-dev-9-2 \
        curl \
        git \
        libcudnn7=7.1.4.18-1+cuda9.2 \
        libcudnn7-dev=7.1.4.18-1+cuda9.2 \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        && \
    rm -rf /var/lib/apt/lists/* && \
    find /usr/local/cuda-9.2/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON python
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.0,3.5,5.2,6.0,6.1
ENV TF_CUDA_VERSION=9.2
ENV TF_CUDNN_VERSION=7

ARG USE_PYTHON_3_NOT_2=True
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ln -s $(which ${PYTHON}) /usr/local/bin/python 

RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    openjdk-8-jdk \
    ${PYTHON}-dev \
    swig

RUN ${PIP} --no-cache-dir install \
    Pillow \
    h5py \
    ipykernel \
    jupyter \
    keras \
    matplotlib \
    mock \
    numpy \
    scipy \
    sklearn \
    pandas \
    && ${PYTHON} -m ipykernel.kernelspec \
    && test "${USE_PYTHON_3_NOT_2}" -eq 1 && true || ${PIP} --no-cache-dir install \
    enum34

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/
COPY run_jupyter.sh /

# Set up Bazel.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc
ENV BAZEL_VERSION 0.11.0
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Download and build TensorFlow.
WORKDIR /tensorflow
RUN git clone --branch=r1.8 --depth=1 https://github.com/tensorflow/tensorflow.git .

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    tensorflow/tools/ci_build/builds/configured GPU \
    bazel build -c opt --copt=-mavx --config=cuda \
	--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        tensorflow/tools/pip_package:build_pip_package && \
    rm /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    pip --no-cache-dir install --upgrade /tmp/pip/tensorflow-*.whl && \
#    rm -rf /tmp/pip && \
    rm -rf /root/.cache

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -q -y tzdata && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

# setup sources.list
RUN echo "deb [arch=amd64,arm64] http://packages.ros.org/ros2/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros2-latest.list
RUN echo "deb [arch=amd64,arm64] http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    ${PYTHON}-rosdep \
    ${PYTHON}-rosinstall \
    ${PYTHON}-vcstools \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init \
    && rosdep update

# install ros1 packages
ENV ROS1_DISTRO melodic
RUN apt-get update && apt-get install -y \
    ros-${ROS1_DISTRO}-desktop=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

# install ros2 packages
ENV ROS2_DISTRO crystal
ENV ROS_MASTER_URI http://localhost:11311
RUN apt-get update && apt-get install -y \
    ros-${ROS2_DISTRO}-desktop=0.6.1-0* \
    # ros-${ROS2_DISTRO}-ros1-bridge=0.6.1-1* \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    ${PYTHON}-colcon-common-extensions \
    ${PYTHON}-argcomplete \
    ${PYTHON}-opencv \
    ${PYTHON}-bson \
    ${PYTHON}-twisted \
    && rm -rf /var/lib/apt/lists/*

# Node.js for ROS2 web bridge
RUN set -ex \
    && curl -sfL https://deb.nodesource.com/setup_9.x | bash - \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs

# ROS2 web bridge
ENV COLCON_PREFIX_PATH /lanefollowing/ros2_ws/install:/opt/ros/${ROS2_DISTRO}
RUN set -ex \
    && cd /opt \
    && git clone https://github.com/RobotWebTools/ros2-web-bridge.git \
    && cd ros2-web-bridge \
    && export CFLAGS="${CFLAGS} -fpermissive" \
    && export CXXFLAGS="${CXXFLAGS} -fpermissive" \
    && bash -c "source /opt/ros/${ROS2_DISTRO}/setup.bash && npm config set python /usr/bin/python2.7 && npm install --global --production --unsafe-perm" \
    && mkdir -p /opt/ros2-web-bridge/node_modules/rclnodejs/generated \
    && chmod 0777 /opt/ros2-web-bridge/node_modules/rclnodejs/generated

ARG USER_NAME=lgsvl

RUN adduser --disabled-password --gecos '' ${USER_NAME} \
    && usermod -aG sudo ${USER_NAME} \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN echo "source /opt/ros/${ROS2_DISTRO}/setup.bash" >> /home/${USER_NAME}/.bashrc
RUN echo "source /lanefollowing/ros2_ws/install/local_setup.bash" >> /home/${USER_NAME}/.bashrc

WORKDIR /lanefollowing
