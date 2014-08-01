function [featuresCell groupCell] = computeFeaturesGlobal( images, sPlabels, siftCentroidsCell, avgClusterHist, nImages, groundTruths )
%Compute features, and ground truth if available, of superpixels for all images

featuresCell = cell( 1,nImages );
computeGT = nargin>5;              %represents if the ground truths are available to be distributed onto all superpixels
if computeGT                       %seperate for loops needed because of parfor with nargin...
    groupCell=cell( 1,nImages );
    parfor i = 1:nImages
        sPlabel = sPlabels{i};
        I = images{i};
        Igt = logical(groundTruths{i});
        Igt = (Igt);
        %calculate features and ground truth of superpixels in image
        [featuresCell{i} groupCell{i}] = computeFeatures( I, sPlabel, siftCentroidsCell{i}, avgClusterHist, Igt ); 
    end
else
    parfor i = 1:nImages
        I = images{i};
        sPlabel = sPlabels{i};
        %calculate features of superpixels in image
        featuresCell{i} = computeFeatures( I, sPlabel, siftCentroidsCell{i}, avgClusterHist );

    end
end

