# Part II Computational Physics Project

### Schr&#246;dinger's Smoke

NOTE: This is a copy of the repository for the Part II Computational Physics project, by dwl38.

Based on the "Schr&#246;dinger's Smoke" problem, which aims to simulate an incompressible and inviscid fluid flow using the technique described by [Chern et al. (2016)](https://cseweb.ucsd.edu/~alchern/projects/SchrodingersSmoke/SchrodingersSmoke.pdf). The report consists only of the `Report.ipynb` notebook, which uses prerendered videos generated from selected test cases.

### About the code

This project was written in an object-oriented style, in order to reduce the workload of planning for and solving each specific sub-problem. The code is scattered amongst the following modules, all contained at the 'ssmoke/' folder level:

* `ssmokeNd` simulates an N-dimensional flow using the *Schr&#246;dinger's Smoke* algorithm.

* `stam2d` and `polytrope2d` simulate a 2-dimensional flow by using two 'na&#239;ve' methods, as explained in the report.

* `dataio` provides an interface for reading/writing data describing the evolution of a fluid flow field over time. It handles both 2D and 3D fields.

* `visualizer` provides an interface for viewing the data files produced by `dataio`, using pre-configured Matplotlib animations.

* `common` provides utility code shared between the other modules.

The test cases used in the report can be found in the 'ssmoke/testcases/' folder; these save the generated data to the 'prerendered/' folder. There are also internal unit tests for certain modules, found in the 'ssmoke/tests/' folder. Both internal tests and testcases should be run from the command line, from the repository root folder, via `python -m ssmoke.tests.*` or `python -m ssmoke.testcases.*` without the .py extension. There is also a rendering script 'ssmoke/RenderAll.py', which renders all data in the 'prerendered/' folder into .mp4 videos and saves them in the 'report/' folder; this script, similarly, is run via `python -m ssmoke.RenderAll`.

A copy of the [FFmpeg](https://ffmpeg.org) executable is included in the repository root folder, in order to generate and save video files. FFmpeg is distributed under the GNU LGPL v2.1 and included in this project as an educational, non-commercial, and non-derivative work.

### References

Chern A., Kn&#246;ppel F., Pinkall U., Schr&#246;der P., Wei&#223;mann S. (2016). Schr&#246;dinger's Smoke. *ACM Transactions on Graphics*, vol. 35, issue 4, article 77, pp. 1-13. DOI: 10.1145/2897824.2925868.
