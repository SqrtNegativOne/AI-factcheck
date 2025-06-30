# AI-factcheck
Crude, simple, and modular AI factchecking implementation.

Scripts, testing files, and research files are in the /scripts/ directory.

Main program can be found at /src/.

Pipeline for old-main.py in the /scripts/ directory: (no chunking method, no dense retrieval or falsifiability checks)
![Pipeline](./meta/diagram.png)

Pipeline for main.py: in progress.

If you'd like to try out different models or algorithms, ideally you should be messing around in the config.py only. The code is modular so should be simple enough.