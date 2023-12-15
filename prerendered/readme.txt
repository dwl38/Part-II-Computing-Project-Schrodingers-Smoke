This folder ('prerendered/') was originally intended to contain compiled data from the testcases.
However, in order to capture the full information of vorticity for the graphical renders, the full velocity field at every timeframe needs to be saved.
This resulted in very large filesizes:

SSmoke2DAirfoil.data:            214 MB
SSmoke2DAirfoilPeriodic.data:    263 MB
SSmoke2DCylinders.data:          287 MB
SSmoke2DStationaryCylinder.data: 337 MB
SSmoke2DVortices.data:           56 MB
SSmoke3DSpheres.data:            8480 MB
SSmoke3DStationarySphere.data:   2890 MB
Stam2DCylinders.data:            143 MB
Stam2DVortices.data:             57 MB

Hence, it was decided (late into the project) not to include these data files into the repository.
Instead they were further processed into video files, found in 'report/'.
The script for this final processing is found in 'ssmoke/RenderAll.py'.

The original data files will be automatically recreated by running the relevant testcases, executed in command line as:
    > python -m ssmoke.testcases.XXX
where XXX is the name of the testcase (without the '.py' extension).