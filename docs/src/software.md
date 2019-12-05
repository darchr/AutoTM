# Software Requirements

AutoTM was developed and tested using Ubuntu 18.04 using [Julia 1.2](https://julialang.org/downloads/oldreleases.html).
We expect it to work on similar operating systems and non-breaking future versions of Julia.

The following are required to build the ngraph dependency:
```
build-essential \
cmake \
clang-6.0 \
clang-format-6.0 \
git \
curl \
zlib1g \
zlib1g-dev \
libtinfo-dev \
unzip \
autoconf \
automake \
libtool
```

## PMM System

If you are running on a system equipped with Intel Optane DC PMMs, the following packages are required
```
numactl \
ipmctl  \
ndctl   
```

## GPU System

If you are using a system with an NVidia GPU, you will additionally need CUDA 10.1/10.2 and cuDNN 7.6.

