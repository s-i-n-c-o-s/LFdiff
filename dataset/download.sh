#!/bin/bash

mkdir -p sig17
cd sig17
wget https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR_Trainingset.zip
wget https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR_Testset.zip
unzip SIGGRAPH17_HDR_Trainingset.zip
unzip SIGGRAPH17_HDR_Testset.zip