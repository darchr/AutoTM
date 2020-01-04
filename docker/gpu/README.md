# AutoTM GPU Dockerfile

Here is a dockerfile with Nvidia/CUDA dependencies that will build all the components required for AutoTM.

## Setup

Make sure `nvidia-docker2` is set-up according the instructions: https://github.com/NVIDIA/nvidia-docker

## Limitations

The [Gurobi](https://www.gurobi.com) ILP solver will not work in Docker containers with the academic license due to licensing limitations (there may be a work-around, but it may be complicated: https://stackoverflow.com/questions/58663540/gurobi-in-docker-container-problem-generating-unique-host-id-for-this-machine.

Fortunately, the fallback Cbc solver still works.
However, note that the build process will generate soft errors about missing Gurobi.
These errors are expected and can be ignored since Gurobi will never be actually used.

**NOTE**: The Cbc solver is significantly less powerful than Gurobi and thus may take longer on problems that Gurobi can easily solve.
If using the Gurobi solver, use the Dockerfile as a template for building the project.

## Building

To build the container, simply execute the build script
```sh
./build.sh
```

## Running the Container

To launch the container, use the script
```sh
./run.sh
```
This will create and map host directories into the docker container to

* Persist kernel profile caches between containers (`caches/`)
* Make experiment data available to the host (`experiment-data/`)
* Make generated figures available to the host (`experiment-=figures/`)

The container is launched in interactive mode.
