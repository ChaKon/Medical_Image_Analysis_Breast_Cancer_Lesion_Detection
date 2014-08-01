function [ images , groundTruths, D , nImages] = loadImages( imagePath, gtPath )

D = dir([imagePath filesep '*.png']);
nImages = numel(D);
images = cell( 1,nImages );
groundTruths = cell( 1,nImages );

for i = 1:nImages
    images{i} = imread( [imagePath filesep D(i).name] );
    groundTruths{i} = logical( imread( [gtPath filesep D(i).name] ) );
end