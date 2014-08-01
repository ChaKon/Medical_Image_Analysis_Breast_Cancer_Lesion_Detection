function [imageSuperPix sPlabels] = computeSuperPixels( images, nImages )
%Compute Quick Shift superpixel images and superpixel labels using
%vl_quickseg. Requires the vl_feat library to have been added to the path.

ratio = 0.2;    %[0,1] tradeoff between color importance and spatial importance (larger values give more importance to color)
kernelsize = 5; % size of the kernel used to estimate the density
maxdist = 10;   % higher: bigger pixels. maximum distance between points in the feature space that may be linked if the density is increased

imageSuperPix = cell( 1,nImages );
sPlabels = cell( 1,nImages );

parfor i = 1:nImages
    I = images{i};
    [imageSuperPix{i} sPlabels{i}] = vl_quickseg(I, ratio, kernelsize, maxdist);

end

