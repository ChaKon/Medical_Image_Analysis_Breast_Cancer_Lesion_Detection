close all;clear all;

warning off;
%loading matlabpool for parfor perfomrance
try 
    matlabpool %try to use parallel for loops
catch err%matlabpool might already be running - parfors will be used
      % or matlabpool might not be possible - it will use standard for loops
end

%loading vlfeat libraries
addVLfeat2path;

%force all steps to be computed even if loaded
forceCompute = false;
tic;

%loading necessary data
imagePath='testingChallange';

%folder where all .mat files of the intermediate steps are stored to speed up computation
trainingTestPath = 'dataChallange';
%folder where results are going to be stored
resultFolder='resultsChallange';

%mat file names of each of the steps
superPixelsMat = 'superPixelsImagesLowRatio.mat';
siftDescriptorsMat = 'siftFeaturesCentroids.mat';
featuresMat = 'featuresMatrix.mat';
clusterHistogramsMat = 'superPixelsClusters50.mat';
fullTrainingMat = 'treeModelFullTraining.mat';
testChallangeResultMat = 'testChallage_Iseg_Ithresh.mat';

if ~exist ( trainingTestPath,'dir' )
    mkdir( trainingTestPath );
end

if ~exist ( resultFolder,'dir' )
    mkdir( resultFolder );
end

D = dir( [imagePath filesep '*.png'] );
nTestingImages=numel(D);

imagesTesting = cell( 1,nTestingImages );
for i = 1:nTestingImages
    imagesTesting{i} = imread( [imagePath filesep D(i).name] );
end

%==========
%computing superpixels of images
if forceCompute || ~exist( [trainingTestPath filesep superPixelsMat],'file' )
    display( 'Computing superpixels...' );
    [imageTestingSuperPix sPlabelsTesting] = computeSuperPixels( imagesTesting, nTestingImages );
    save( [trainingTestPath filesep superPixelsMat],'imageTestingSuperPix','sPlabelsTesting' );
else
    display( ['Loading superpixels from file: ' superPixelsMat] );
    load(  [trainingTestPath filesep superPixelsMat] );
end
%end of superpixels

%==========
%computing SIFT descriptors on superpixel centroids
if forceCompute || ~exist( [trainingTestPath filesep siftDescriptorsMat],'file' )
    display( 'Computing Sift descriptors...' );
    sPsiftFeatures = computeSiftCentroids( imagesTesting,sPlabelsTesting, nTestingImages );
    save( [trainingTestPath filesep siftDescriptorsMat],'sPsiftFeatures');
else
    display( ['Loading Sift descriptors from file: ' siftDescriptorsMat] );
    load( [trainingTestPath filesep siftDescriptorsMat] );
end
%end of computing SIFT descriptors on superpixel centroids

%==========
%loading the cluster histograms of the superpixels
display( ['Loading cluster histograms of superpixels: ' clusterHistogramsMat] );
load( [trainingTestPath filesep clusterHistogramsMat] );

%==========
%computing features
if forceCompute || ~exist( [trainingTestPath filesep featuresMat],'file' )
    display( 'Computing feature Matrix...' );
    featuresCellTesting = computeFeaturesGlobal( imagesTesting, sPlabelsTesting, sPsiftFeatures, clusterAvgHist, nTestingImages );
    save([trainingTestPath filesep featuresMat], 'featuresCellTesting');
else
    display( ['Loading features matrix from file: ' featuresMat] );
    load([trainingTestPath filesep featuresMat]);
end
%end of computing features

%==========
%loading training model
load( [trainingTestPath filesep fullTrainingMat ]);

%==========

%start the testing with random forest
testImagesSegBefore = cell( 1,nTestingImages );
testImagesSeg = cell( 1,nTestingImages );
testImagesThres = cell( 1,nTestingImages );
parfor i = 1:nTestingImages
    test = featuresCellTesting{i};              %features of superpixels to be classified
    outputClassifier = predict(treeModelFullTraining,test); %predict probability of each superpixel as lesion, using the model trained without its image
    Iseg = outputClassifier(sPlabelsTesting{i});       %copy superpixels' probabilities to all pixels within
    Iseg = Iseg-min(Iseg(:));                   %make probability map have a min of 0
    Iseg = Iseg/max(Iseg(:));                   %make probability map have a max of 1
    testImagesSegBefore{i} = Iseg;              %store output of classifier
    
    %parameters from optimiser output
    params = [0.0431    0.5503    0.1236    0.3005    0.5167    0.2360    0.5813    0.3445];
    
    %apply post-processing of reducing the probability of superpixels far from [x_0,y_0]
    sigma_x=params(5);     sigma_y=params(6);      x_0=params(7);     y_0=params(8);
    Iseg = updateProbs_center( Iseg, sPlabelsTesting{i},sigma_x,sigma_y,x_0,y_0 );
    testImagesSeg{i} = Iseg;                    %store output of post-processing
    
    %threshold the image to determine the final segmentation
    a = params(1);      b = params(2);      c = params(3);      d = params(4);
    threshValue = min(max(Iseg(:))-a*std(outputClassifier),b)*c + d; %determine the relevant threshold
    Ithresh = Iseg>threshValue;                 %threshold post-processed probability map
    testImagesThres{i} = logical(Ithresh);      %store final segmentation
    
end
save( [trainingTestPath filesep testChallangeResultMat],'testImagesThres','testImagesSeg' );
toc

%==========
%ploting output images of each stage
for i=1:nTestingImages
    figure();
    subplot(221);imshow( imagesTesting{i}, [] );    
    subplot(222);imshow( testImagesSegBefore{i}, [] );
    subplot(223);imshow( testImagesSeg{i}, [] );
    subplot(224);imshow( testImagesThres{i} );
    
end

%==========
%Saving the results
nImages=numel(testImagesThres);
for i=1:nImages
    I=testImagesThres{i};  
    imwrite( I,[resultFolder filesep 's' D(i).name] );
end