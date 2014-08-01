function siftCentroidsCell=computeSiftCentroids( images,sPlabels, nImages )
%Calculate SIFT descriptor on the centroid of each superpixel
siftCentroidsCell=cell(1,nImages);

parfor i=1:nImages
    I=images{i};
    sPstats = regionprops(sPlabels{i},I,'Centroid');    %calculate the centroid of each superpixel in image
    numSP = size(sPstats,1);                            %number of superpixels in the image
    sPcenters = reshape( [sPstats.Centroid],2, numSP)'; %extract the centroid of the superpixels
    sPsiftFeatures = zeros( numSP, 128 );
    for j=1:numSP 
        midPoint = floor(fliplr(sPcenters(j,:)));       %convert centroid from [x,y] to [i,j]
        bboxSift = [midPoint-5,midPoint+5];             %calculate bounding box around centroid
        
        try                                             %calculate SIFT descriptor on centroid 
            [~, siftFeature] = vl_dsift( single( I ),'bounds',bboxSift,'step',10 );
            sPsiftFeatures(j,:) = siftFeature(:,1)';
        catch err
            sPsiftFeatures(j,:) = zeros(1,128);         %boundary superpixels can fail
        end
    end
    siftCentroidsCell{i} = sPsiftFeatures;              %store image's descriptors
end
