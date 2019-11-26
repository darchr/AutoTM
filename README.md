# AutoTM

Documentation coming soon™!

## Sub Projects

| Repo | Link   | Description |
|------|--------|-------------|
| ngraph (fork)       | https://github.com/darchr/ngraph/tree/mh/pmem  | Customized fork of ngraph source code |
| nGraph.jl           | https://github.com/hildebrandmw/nGraph.jl      | Julia frontend for nGraph             |
| PersistentArrays.jl | https://github.com/darchr/PersistentArrays.jl  | NVDIMM backed arrays |
| SystemSnoop.jl      | https://github.com/hildebrandmw/SystemSnoop.jl | Base System monitoring API |
| PCM.jl              | https://github.com/hildebrandmw/PCM.jl         | Wrapper for Intel [pcm](https://github.com/opcm/pcm)  |
| pcm                 | https://github.com/hildebrandmw/pcm | Customized fork of Intel pcm |
| MaxLFSR.jl          | https://github.com/hildebrandmw/MaxLFSR.jl | Maximum length Linear Feedback Shift Registers |

## Saving and Loading Experimental Data

All of the code for AutoTM, including code for experiments, is version controlled in various Git repositories.
However, results from experiments, which can include serialized Julia data structures as well as figures, are not version controlled due to size limitations.
Raw data for the `Benchmarker` suite of experiments exceeds 100 MB.
There is a mechanism to save and load all experimental data.

### Saving

To save experimental data, navigate to the `scripts/` folder and run the command
```
julia saver.jl save
```
This will gather all of the relevant experimental data, as well as the kernel profile cache and bundle it into the tarball `autotm_backup.tar.gz`.

### Loading
To reverse this process, run
```
julia saver.jl load --tarball <path/to/tarball> [--force]
```
This will unpack the tarball and put all the contents back where they came from.
The `--force` flag will remove existing directories and replace them with the tarball data.
