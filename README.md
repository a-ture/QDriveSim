# AutoGuidaAutonoma
Progetto per il Corso di FVAB



## Requirements

### Software: 
- Python: 3.8
- CARLA: 0.9.15
- Libraries: install from `requirements.txt`


### System requirements: 
- GPU: the server needs at least a 6 GB GPU although 8 GB is recommended. A dedicated GPU is highly recommended for machine learning.
- RAM: at least 16 or 32 Gb.
- x64 system: The simulator should run in any 64 bits Windows system.
- 165 GB disk space. CARLA itself will take around 32 GB and the related major software installations (including Unreal Engine) will take around 133 GB.

### Minor installations
- **CMake** generates standard build files from simple configuration files. It's recommended you use version 3.15+.
- **Make** generates the executables. It is necessary to use Make version 3.81, otherwise the build may fail. If you have multiple versions of Make installed, check that you are using version 3.81 in your PATH when building CARLA. You can check your default version of Make by running make --version.
- **7Zip** is a file compression software. This is required for automatic decompression of asset files and prevents errors during build time due to large files being extracted incorrectly or partially.
-  **pip** version 20.3 or higher is required.

## Installation

Before running any code from this repo you have to:
1. **Clone this repo**: `git clone https://github.com/a-ture/AutoGuidaAutonoma`
2. **Download CARLA 0.9.15** from their GitHub repo, [here](https://github.com/carla-simulator/carla/releases/tag/0.9.15/) 
   where you can find precompiled binaries which are ready-to-use. Refer to [carla-quickstart](https://carla.readthedocs.io/en/latest/start_quickstart/)
   for more information.
3. **Install CARLA Python bindings** in order to be able to manage CARLA from Python code. Make sure to use Anaconda Virtual Envoriment, open your terminal and type:
```
 pip3 install carla
```

4. Before running the repository's code be sure to start CARLA first:
- Windows: your-path-to/CARLA_0.9.15/WindowsNoEditor/CarlaUE4.exe
- Linux: your-path-to/CARLA_0.9.15/./CarlaUE4.sh
- [optional] To use less resources add these flags to the previous command: -windowed -ResX=32 -ResY=32 --quality-level=Low. For example ./CarlaUE4.sh --quality-level=Low.



   


