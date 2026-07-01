#### Example script

Following is an example script that allows to fit a plane on Analysis points created in Mimcs. The script calls another function which performs least square fitting on the array of input points. You will need the �fit_3d_data.m� function to run this which is downloaded with this package. Note that to call an external function, first the *.m file should be added to the path. This is done by using the addpath command in Matlab. 

%% Script information

% Author: Nikhil Sindhwani

% Date: 20/03/2012

% Copyright 2012: Materialise NV

% Function Description: This function fits a plane on given set of points and creates a Mimics object

% Dependencies: This script calls the "fit_3D_data.m" function from Matlab central. 

% The function " fit_3D_data.m" can be downloaded here: http://uc.materialise.com/mimics/node/1236 

%% Script begin

% Mimics.Input: input1

% Mimics.Output: output1

% n = number of Analysis points in the input

n = size(input1);

%%Plane fitting

% Create matrix for points used for plane fitting

for i = 1 : n(2)

X (i,1) = input1{i}.Points(1);

Y (i,1) = input1{i}.Points(2);

Z (i,1) = input1{i}.Points(3);

end

addpath C:\path_to_fit_3D_data_function

[err, N, P] = fit_3D_data( X, Y, Z, 'plane', 'on', 'on')

% Create Mimics Object

output1 = struct('Type', 'Mimics.Analysis.Plane', 'Name', 'fitplane','Points', P, 'Normal', N', 'Width', 100, 'Height', 100);

%% Script end
