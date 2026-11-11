# BORIS build requirements on the Modal image

`Spin_Colour_Kernel.boris_pair.build_boris` clones BORIS (https://github.com/SerbanL/BORIS) at
`048503d2ca6475e91b073eb1762ecf9d3634c3e1` and runs its makefile unchanged:

```
make configure arch=<major>0 sprec=<1 single | 0 double> python=<system 3.X> cuda=<nvcc release>
make compile -j <cpus>
make install            # links BorisLin in the clone root
```

One clone per precision and architecture (`/tmp/sck/boris-sp-arch80`, `boris-dp-arch90`, ...); a cached
`BorisLin` is reused. `arch` is the compute-capability major times ten, because BORIS selects a GPU only when
`major * 100 == __CUDA_ARCH__` (Boris/Simulation.cpp) and `configure` writes `__CUDA_ARCH__ = arch * 10`.

## What the makefile hard-codes

| Need | Where | Package (Ubuntu 22.04) |
|---|---|---|
| g++ with OpenMP, GNU make 4 (`$(file ...)`) | compile and link | `build-essential` (`libomp-dev` per the README) |
| nvcc, cufft, cudart, cudadevrt, NVML stub and `nvml.h` under `/usr/local/cuda-<ver>/targets/x86_64-linux/` | all `.cu`, link | the `nvidia/cuda:*-devel` base image (already used) |
| `fftw3.h`, `/usr/lib/x86_64-linux-gnu/libfftw3.so.3` | ConvolutionData.h, link | `libfftw3-dev` |
| `-ltbb` | link | `libtbb-dev` |
| `-lX11` (`X11/Xlib.h`) | link, one include | `libx11-dev` |
| `/usr/include/python3.X/Python.h`, `-lpython3.X` | embedded Python | `python3-dev` (system Python 3.10, not Modal's `add_python` build) |
| `libnvidia-ml.so.1` at run time | NVML | provided by the Modal GPU driver |

## Lines to add to `remote/modal_h100.py`

The current image already has nvcc (`nvidia/cuda:12.4.1-devel-ubuntu22.04`) and `git`, `build-essential`. Add, after
the existing `.apt_install("git", "build-essential")`:

```python
    .apt_install("libomp-dev", "libtbb-dev", "libfftw3-dev", "libx11-dev", "python3-dev")
```

No `.run_commands` line is needed: the build runs at run time inside the experiment, like VAMPIRE's.

## Run time

`BorisLin -g <device> -s <abs path>/pair.py` runs the generated embedded-Python script headless (CompileFlags.h
already sets `GRAPHICS 0`) and exits with SIGTERM (return code -15) when the script ends. The script uses only the
`BPython` module BORIS embeds and the standard library, so the system Python needs no numpy or matplotlib.
`PYTHONHOME` and `PYTHONPATH` are removed from BORIS's environment so the embedded 3.10 interpreter finds its own
standard library. A full build of one precision compiles several hundred C++ and CUDA files; expect tens of minutes
per precision on the first run of a container.
