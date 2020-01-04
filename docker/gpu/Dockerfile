FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG WHOAMI=user

# Setup build as current user to Gurobi works correctly
RUN groupadd -g ${GROUP_ID} ${WHOAMI} && \
    useradd -l -d /home/${WHOAMI} -u ${USER_ID} -g ${WHOAMI} ${WHOAMI} && \
    install -d -m 0755 -o ${WHOAMI} -g ${WHOAMI} /home/${WHOAMI}

# Set the GUROBI_HOME variable to "/gurobi"
#
# Due to licensing, we can't build Gurobi into the container and must instead supply it
# at run time.
#
# Here, we just create a dummy directory.
#
# This will create errors while building the Gurobi package, but those will have to be fixed
# after container instantiation.
# RUN mkdir /home/${WHOAMI}/gurobi
# ENV GUROBI_HOME="/home/${WHOAMI}/gurobi"

# Install deps
RUN apt-get update && apt-get install -y \
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

# Install Julia at /julia
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.2/julia-1.2.0-linux-x86_64.tar.gz
RUN tar -xzf julia-1.2.0-linux-x86_64.tar.gz -C /home/${WHOAMI}
ENV PATH="$PATH:/home/${WHOAMI}/julia-1.2.0/bin"

### Install AutoTM

# At the moment, we're cloning the GIT repo master.
#
# For the final artifact, we can fetch the tarball/zip file directly from the DOI
RUN git clone --recursive https://github.com/darchr/AutoTM /home/${WHOAMI}/AutoTM
ENV AUTOTM_HOME="/home/${WHOAMI}/AutoTM"
WORKDIR /home/${WHOAMI}/AutoTM

# Setup build parameters for GPU - no Gurobi
RUN echo '{"DEBUG":false,"PMDK":false,"NUMA":false,"GPU":true}' > deps/nGraph/deps/build.json
RUN echo '{"GUROBI":false}' > AutoTM/deps/build.json

# Switch to new user to build and instantiate project.
RUN chown --silent --recursive ${USER_ID}:${GROUP_ID} /home/${WHOAMI}
USER ${WHOAMI}

# Build Benchmarks
WORKDIR /home/${WHOAMI}/AutoTM/experiments/Benchmarker
RUN julia --color=yes --project -E 'using Pkg; Pkg.instantiate()'
RUN julia --color=yes --project -E 'using Pkg; Pkg.build(; verbose = true); using Benchmarker, AutoTM'

