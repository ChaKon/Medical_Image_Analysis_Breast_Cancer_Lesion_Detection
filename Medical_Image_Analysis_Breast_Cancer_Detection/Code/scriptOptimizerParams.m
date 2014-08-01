close all; clear all;
%script that computes the optimal configuration of parameters to set the
%best threshold and post processing function parameters

%global variables are needed as objective function does not accept extra parameters
global groundTruths
global testedImages
global nImages
global sPlabels
global deviations

trainingTestPath = 'data_leave_one_out';
imagePath = 'training_testing';
gtPath = 'training_testing/GT';

superPixelsMat = 'superPixelsImagesLowRatio.mat';
testingMat = 'testing.mat';

load( [trainingTestPath filesep superPixelsMat] );
load( [trainingTestPath filesep testingMat] );

[ ~ , groundTruths, ~ , nImages] = loadImages( imagePath, gtPath );
testedImages=testImagesSeg;

for i=1:nImages
Iseg=testedImages{i};

deviations(i)=std(unique(Iseg)); 
end

%initial parameters
a = 1;
b = 0.55;
c = 1;
d = 0;
sigma_x =0.3;
sigma_y =0.3;
x_0 =0.5;
y_0 =0.3;
params0 = [a b c d sigma_x sigma_y x_0 y_0];

%limits on parameters
lowerBound = [0 0 0 0 0   0   0    0];
upperBound = [1 1 1 1 0.6 0.6 0.65 0.65];

%compute optimisation
options = saoptimset('MaxIt', 2000,'Display','iter'); %settings for maximum iterations and when to display
tic;
[x,fval,exitflag] = simulannealbnd(@diceCoefObjectiveFunctionRandomForest,params0, lowerBound,upperBound,options)
toc