# Software Requirements

AutoTM was developed and tested using Ubuntu 18.04 using [Julia 1.2](https://julialang.org/downloads/oldreleases.html).
We expect it to work on similar operating systems and non-breaking future versions of Julia.

The following are required to build the ngraph dependency:
```
autoconf \
automake \
build-essential \
cmake \
clang \
clang-format \
curl \
git \
libdaxctl-dev \
libndctl-dev \
libtinfo-dev \
libtool \
ndctl \
pkg-config \
unzip \
wget \
zlib1g \
zlib1g-dev
```

## GPU System

If you are using a system with an NVidia GPU, you will additionally need CUDA 10.1/10.2 and cuDNN 7.6.

