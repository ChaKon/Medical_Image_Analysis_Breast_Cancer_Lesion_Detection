function refinedProbs = updateProbs_center( probs,sPlabel,sigma_x,sigma_y,x_0,y_0)
%Refine probability map 'probs' of superpixels in 'sPlabel' using inverse 
%elliptical bell curve centered at [x_0,y_0], with shape determined by sigmas
%Initial paramaters which might be a good intial guess: [0.2,0.2,0.5,0.3]

sPstats = regionprops(sPlabel,'Centroid');          %calculate centroid of each superpixel
numSP = size(sPstats,1);                            %number of superpixels

sPcenters = reshape( [sPstats.Centroid],2, numSP)'; %extract centroids
sPcentersN = sPcenters./repmat([size(sPlabel,2), size(sPlabel,1)],numSP,1); %normalise centroids by image dimensions

refinedProbs=probs;
for j=1:numSP
    %calculate x- and y-distance between normalised centroid and the center
    distToCenter = [abs(sPcentersN(j,1)-x_0),abs(sPcentersN(j,2)-y_0)]; 
    %calculate the minimum of the weights from the x and y distance
    weight = min(exp(-distToCenter(1)^2/2/sigma_x^2), exp(-distToCenter(2)^2/2/sigma_y^2));
    %update probability map using weights
    refinedProbs(sPlabel==j) = probs(sPlabel==j)*weight; %update entire superpixel's value
end

%scale probability map so in [0,1];
refinedProbs = refinedProbs - min(refinedProbs(:));
refinedProbs = refinedProbs/max(refinedProbs(:));


