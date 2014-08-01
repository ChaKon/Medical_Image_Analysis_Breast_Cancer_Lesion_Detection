function [features sPgroundTruth] = computeFeatures(I, sPlabels, siftFeaturesCentroid, clusterHistograms, Igt)
%Compute and concatenate features for all superpixels, and ground truth if available

%exract properties of superpixel regions
sPstats = regionprops(sPlabels,I,'Centroid','Area','MeanIntensity','PixelValues','BoundingBox');

%calculate superpixel features
[sPglcmContrast,sPglcmCorrelation,sPglcmEnergy,sPglcmHomogeneity]=textureFeatures(sPstats,sPlabels,I);
[sPmean,sPmedian,sPstdev,sPdistsToHists]=intensityFeatures(sPstats,clusterHistograms);
[sPcenters,sPareas,sPdistsToCenter]=spatialFeatures(sPstats,size(I));
sPsiftFeatures = siftFeaturesCentroid;

%concatenate feature vectors
features = [sPglcmContrast,sPglcmCorrelation,sPglcmEnergy,sPglcmHomogeneity,... %texture features
            sPmean, sPmedian, sPstdev, sPdistsToHists,...%intensity features
            sPdistsToCenter,...                          %spatial features
            sPsiftFeatures];                             %SIFT features

%calculate ground truth for each superpixel
if nargin>4 %if ground truth is available
    centerIdxs = [round(sPcenters(:,2)),round(sPcenters(:,1))];
    centerIdxsLin = sub2ind(size(Igt),centerIdxs(:,1),centerIdxs(:,2));
    sPgroundTruth = Igt(centerIdxsLin);
else
    sPgroundTruth = [];
end
end


%--------------------------------------------------------- TEXTURE FEATURES
function [sPglcmContrast,sPglcmCorrelation,sPglcmEnergy,sPglcmHomogeneity]=textureFeatures(sPstats,sPlabels,I)
%Calculate texture features from GLCM

dists_set = 1:6; %distances to be used for GLCM offsets
angles_set = [0 1; -1 1; -1 0; -1 -1]; %0, 45, 90, 135 degrees offsets
numLevels = 15; %for GLCM matrix calculation (one level will be discarded)

%calculate set of offsets given angle and distance sets
offsets = []; 
for i=1:length(dists_set) %append angle set scaled by each distance
    offsets = [offsets; angles_set.*dists_set(i)]; 
end

%initialise variables
numSP = size(sPstats,1);
sPglcmEnergy = zeros( numSP,size(offsets,1) );
sPglcmContrast = zeros( numSP,size(offsets,1) );
sPglcmCorrelation = zeros( numSP,size(offsets,1) );
sPglcmHomogeneity = zeros( numSP,size(offsets,1) );

for j=1:numSP
    %compute graycomatrix for each region
    bbox = round(sPstats(j).BoundingBox); %extract bounding box of SP
    mask = sPlabels==j;                    %determine mask of SP pixels within bounding box
    mask = mask( bbox(2):(bbox(2)+bbox(4)-1),bbox(1):(bbox(1)+bbox(3)-1) );
    subI = I( bbox(2):(bbox(2)+bbox(4)-1),bbox(1):(bbox(1)+bbox(3)-1) );
    mask = uint8(mask);
    subI = subI.*mask;
    mask(mask==0) = 255;                  %set pixels in bounding box which aren't SP to max
    mask(mask==1) = 0;
    subI = subI + uint8(mask);
    
    %get features for all offsets
    for i=1:size(offsets,1)
        try
            glcm =graycomatrix( subI,'NumLevels',numLevels,'Offset',offsets(i,:),'Symmetric',true );
        catch err                   %in case error occurs at the boundaries
            glcm =zeros(2);
            err
        end
        glcm=glcm(1:end-1,1:end-1); %ignore last row and column, where artifacts of non-SP pixels are
        
        %calculate and store statistics
        statsGlcm = graycoprops(glcm, {'Contrast','Correlation','Energy','Homogeneity'});
        sPglcmContrast(j,i) = statsGlcm.Contrast;
        sPglcmCorrelation(j,i) = statsGlcm.Correlation;
        sPglcmEnergy(j,i) = statsGlcm.Energy;
        sPglcmHomogeneity(j,i) = statsGlcm.Homogeneity;
    end
end
end


%------------------------------------------------------- INTENSITY FEATURES
function [sPmean,sPmedian,sPstdev,sPdistsToHists]=intensityFeatures(sPstats,clusterHistograms)
%Calculate intensity features from PixelValues and Chi^2 dists to cluster histograms

nBins = 10;                         %used for generating histogram bins
[~,binCenters] = hist(0:255,nBins); %calculate histogram bin centers
numSP = size(sPstats,1);
sPhist = zeros(numSP,nBins);
sPmean = zeros(numSP,1);
sPmedian = zeros(numSP,1);
sPstdev = zeros(numSP,1);
numClusters = size(clusterHistograms,1);
sPdistsToHists = zeros(numSP,numClusters);

for j=1:numSP
    %extract intensity values withing the SP from sPstats
    sPvals = double(reshape( [sPstats(j).PixelValues],1,length(sPstats(j).PixelValues)));
    sPmean(j) = mean(sPvals);       %calculate mean of intensities in SP
    sPmedian(j) = median(sPvals);   %calculate mean of intensities in SP
    sPstdev(j) = std(sPvals);       %calculate std deviation of intensities in SP
    sPhist_ = hist(sPvals,binCenters); %calculate histogram of intensities
    sPhist(j,:) = sPhist_/trapz(binCenters,sPhist_); %normalise histogram
    for i=1:numClusters             %calculate chi-squared distance to each cluster histogram
        sPdistsToHists(j,i)=distChiSq(sPhist(j,:),clusterHistograms(i,:));
    end
end
end


%--------------------------------------------------------- SPATIAL FEATURES
function [sPcenters,sPareas,sPdistsToCenter]=spatialFeatures(sPstats,imSize)
%Calculate spatial features of superpixels

numSP = size(sPstats,1);
sPcenters = reshape( [sPstats.Centroid], 2, numSP)';            %extract centroids of SPs
sPcentersN = sPcenters./repmat([imSize(2), imSize(1)],numSP,1); %normalise by image dimensions
sPareas = reshape( [sPstats.Area], 1, numSP)';                  %extract areas of SPs
sPareas = sPareas./repmat(imSize(2)*imSize(1),numSP,1);         %normalise by image area

%calculate the distance from SP centroids to center of image
sPdistsToCenter = repmat([0.5 0.], numSP,1) - sPcentersN;
sPdistsToCenter = sqrt( sum(sPdistsToCenter.*sPdistsToCenter,2) );
end


%----------------------------------------------------- Chi squared distance
%Taken from http://www.cs.columbia.edu/~mmerler/project/code/pdist2.m
function D = distChiSq( X, Y )
m = size(X,1);  n = size(Y,1);
mOnes = ones(1,m); D = zeros(m,n);
for i=1:n
    yi = Y(i,:);  yiRep = yi( mOnes, : );
    s = yiRep + X;    d = yiRep - X;
    D(:,i) = sum( d.^2 ./ (s+eps), 2 );
end
D = D/2;
end
