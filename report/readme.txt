This folder ('report/') contains images and videos to be directly linked within the report.

The report was originally intended to render 'live' illustrations from precalculated data within the 'prerendered/' folder.
However, the very large filesizes of the precalculated data made this infeasible.
Hence, it was decided (late into the project) not to include the data files into the repository.
Instead they were further processed into video files, found in this folder.
The script for this final processing is found in 'ssmoke/RenderAll.py'.

The rendering script may be executed in command line as:
    > python -m ssmoke.RenderAll
from the root folder.