close all; clear all;

%paths for storing and loading data
imagePath = 'training_testing';
gtPath = 'training_testing/GT';
trainingTestPath = 'data_leave_one_out';
outputPath = 'dataChallange';

%mat files with the necessary data
fullTrainingMat = 'treeModelFullTraining.mat';
featuresMat = 'featuresMatrix.mat';
superPixelsMat = 'superPixelsImagesLowRatio.mat';
siftDescriptorsMat = 'siftFeaturesCentroids.mat';

if ~exist ( trainingTestPath, 'dir' )
    mkdir( trainingTestPath );
end

%loading images and grounturth in cell arrays
[ images , groundTruths, imageNames , nImages] = loadImages( imagePath, gtPath );

display( ['Loading superpixels from file: ' superPixelsMat] );
load( [trainingTestPath filesep superPixelsMat] );
display( ['Loading Sift descriptors from file: ' siftDescriptorsMat] );
load( [trainingTestPath filesep siftDescriptorsMat] );

%checking if features of the training set were computed or not
featureStatus=exist([trainingTestPath filesep featuresMat],'file');
if ~featureStatus
    %computing features and saving them
    display( 'Computing feature Matrix...' );
    [featuresCell groupCell] = computeFeaturesGlobal( images, sPlabels, siftCentroidsCell, clusterAvgHist, nImages, groundTruths );
    save( [trainingTestPath filesep featuresMat] , 'featuresCell', 'groupCell' );
else 
    display( ['Loading features matrix from file: ' featuresMat] );
    load( [trainingTestPath filesep featuresMat] );
end

%freeing memory as there is a lot of data to compute
clear images;
clear groundTruths;
clear sPlabels;
clear siftCentroidsCell;
clear imageSuperPix;
pack
%Full training
display( 'Computing Full Training...' );

maskTrain=logical((1:nImages)*0); %train with all the images images

%cannot be run because the laptop has not enough memory
%treeModelFullTraining=TreeBagger(80,training,double(group),'Method','regression','MinLeaf',5,'Fboot',1);

%breaking the training in two trees as our computers do not have enough
%memory to handle all the data
maskTrain( 1:floor(nImages/2) )=1;
training=cell2mat( featuresCell(maskTrain)' );
group=cell2mat( groupCell(maskTrain)' );
b1 = TreeBagger( 80, training, double(group), 'Method', 'regression', 'MinLeaf', 5, 'Fboot', 1 );

maskTrain=~maskTrain;
training=cell2mat( featuresCell(maskTrain)' );
group=cell2mat( groupCell(maskTrain)' );
b2 = TreeBagger( 80, training, double(group), 'Method', 'regression', 'MinLeaf', 5, 'Fboot', 1 );
c1 = compact( b1 );
c2 = compact( b2 );

%merging the trees to have the full training model
treeModelFullTraining = combine( c1, c2 );


save( [outputPath filesep fullTrainingMat], 'treeModelFullTraining' );



