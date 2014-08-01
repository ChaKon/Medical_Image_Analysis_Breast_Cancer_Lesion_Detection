close all;
clear all;

warning off;
%loading matlabpool for parfor perfomrance
try 
    matlabpool %try to use parallel for loops
catch err %matlabpool might already be running - parallel for-loops will be used
      % or matlabpool might not be possible - standard for-loops will be used
end

%loading vlfeat libraries
addVLfeat2path;
%paths of images and ground truth
imagePath = 'training_testing';
gtPath = 'training_testing/GT';
resultsPath= 'resultsLeaveOneOut';

%boolean variables for forcing to recompute any step of the pipeline
forceComputeSuperPixel = false;
forceComputeSiftDescriptors = false;
forceComputeFeatures = false;
forceComputeClusterHistograms = false;
forceComputeTraining = false;
forceComputeTesting = true;

%mat file names for each step if the pipeline
superPixelsMat = 'superPixelsImagesLowRatio.mat';
siftDescriptorsMat = 'siftFeaturesCentroids.mat';
featuresMat = 'featuresMatrix.mat';
clusterHistogramsMat = 'superPixelsClusters50.mat';
trainingMat = 'training.mat';
testingMat = 'testing.mat';

%folder where all .mat files of the intermediate steps are stored to speed up computation
trainingTestPath = 'data_leave_one_out';

if ~exist ( trainingTestPath,'dir' )
    mkdir( trainingTestPath );
end

%loading images and grountruth in cell arrays
[ images, groundTruths, imageNames, nImages] = loadImages( imagePath, gtPath );

%computing or loading superpixels of images
if forceComputeSuperPixel || ~exist( [trainingTestPath filesep superPixelsMat],'file' )
    display( 'Computing superpixels...' );
    [imageSuperPix sPlabels] = computeSuperPixels( images, nImages ); %oversegment images into superpixels
    save([trainingTestPath filesep superPixelsMat],'imageSuperPix','sPlabels');
else
    display( ['Loading superpixels from file: ' superPixelsMat] );
    load( [trainingTestPath filesep superPixelsMat] );
    
end

%computing or loading SIFT descriptors of the center of the superpixels
if forceComputeSiftDescriptors || ~exist( [trainingTestPath filesep siftDescriptorsMat],'file' )
    display( 'Computing Sift descriptors...' );
    siftCentroidsCell = computeSiftCentroids( images,sPlabels, nImages ); %calculate SIFT descriptors on SP centroids
    save( [trainingTestPath filesep siftDescriptorsMat],'siftCentroidsCell');
else
    display( ['Loading Sift descriptors from file: ' siftDescriptorsMat] );
    load( [trainingTestPath filesep siftDescriptorsMat] );
end

%computing or loading cluster histograms of superpixels
if forceComputeClusterHistograms || ~exist( [trainingTestPath filesep clusterHistogramsMat],'file' )
    display( 'Computing cluster histograms of superpixels...' );
    numClusters = 50;
    clusterAvgHist = clusterSuperPixels(images,sPlabels,numClusters); %cluster superpixels based on histograms
    save( [trainingTestPath filesep clusterHistogramsMat],'clusterAvgHist', 'numClusters' );
else
    display( ['Loading cluster histograms of superpixels: ' clusterHistogramsMat] );
    load( [trainingTestPath filesep clusterHistogramsMat] );
end

%computing or loading of features and ground truths for each superpixel
if forceComputeFeatures || ~exist( [trainingTestPath filesep featuresMat],'file' )
    display( 'Computing feature matrix...' );
    [featuresCell groupCell] = computeFeaturesGlobal( images, sPlabels, siftCentroidsCell, clusterAvgHist, nImages, groundTruths );
    save( [trainingTestPath filesep featuresMat] , 'featuresCell', 'groupCell' );
else 
    display( ['Loading features matrix from file: ' featuresMat] );
    load( [trainingTestPath filesep featuresMat] );
end

%packing matlab memory to have more free space
pack();

%computing or loading of leave-one-out training information
if forceComputeTraining || ~exist( [trainingTestPath filesep trainingMat],'file' )
    display( 'Computing leave-one-out training information...' );
    %train a classifier for each image which excludes it from the training set
    treeModels = computeTrainingLeaveOneOut( featuresCell, groupCell, nImages ); 
    save( [trainingTestPath filesep trainingMat] , 'treeModels' );
else
    display( ['Loading training information Leave one out from file: ' trainingMat] );
    load( [trainingTestPath filesep trainingMat] );  
end

%computing or loading of leave-one-out testing
if forceComputeTesting || ~exist( [trainingTestPath filesep testingMat],'file' )
    display( 'Computing leave-one-out testing...' );
    %test each image with the classifier which was trained by excluding it from the training set
    [testImagesThres testImagesSegUpdate testImagesSeg evaluationResults] = computeTestingLeaveOneOut( featuresCell, sPlabels, treeModels, groundTruths, nImages );
    save( [trainingTestPath filesep testingMat] , 'testImagesThres', 'testImagesSegUpdate', 'testImagesSeg', 'evaluationResults' );
else
    display( ['Loading leave-one-out testing from file: ' testingMat] );
    load( [trainingTestPath filesep testingMat] );  
end

%ploting the output results and saving images to results folder
verbose=true; %if verbose then plot the output of all of the images otherwise plot only the box diagram of the evaluation results
displayOutputLeaveOneOut( testImagesThres, testImagesSegUpdate, testImagesSeg, evaluationResults, images, groundTruths, nImages,verbose, resultsPath, imageNames );

