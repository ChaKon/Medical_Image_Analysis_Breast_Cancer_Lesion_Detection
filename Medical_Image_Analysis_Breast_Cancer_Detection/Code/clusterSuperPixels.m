function clusterAvgHist = clusterSuperPixels(images,sPlabels,numClusters)
%Perform clustering on superpixels based on Chi-squared distance to other
%superpixels' normalised histograms

sPhists = calculateHistograms(images,sPlabels);              %calculate the histograms of all superpixels in all images
cluster_idxs = clusterHistograms(sPhists,numClusters);       %cluster histograms on chi-squared distance to others

%determine average histogram of each cluster
clusterAvgHist=zeros(numClusters,size(sPhists,2));
for i=1:numClusters
    SPsInCluster = find(cluster_idxs==i);                    %extract superpixels belonging to cluster
    clusterAvgHist(i,:) = sum(sPhists(SPsInCluster,:),1)/length(SPsInCluster); %average the relevant histograms
end

end


%----------------------------------------------------- CALCULATE HISTOGRAMS
function sPhists = calculateHistograms(images,sPlabels)
%Calculate the histograms of all superpixels in all images on the
%chi-squared distance to all others
 
numImgs = size(images,2);
nBins = 10;                         %used for generating histogram bins only
[~,binCenters] = hist(0:255,nBins); %calculate histogram bin centers
sPhists = [];

for i=1:numImgs
    sPstats = regionprops(sPlabels{i},images{i},'PixelValues'); 
    sPhistsOfImg = [];
    numSP = length(sPstats);
    for j=1:numSP
        %extract intensity values withing the SP from sPstats
        sPvals = double(reshape( [sPstats(j).PixelValues],1,length(sPstats(j).PixelValues)));
        sPhist = hist(sPvals,binCenters);                     %calculate histogram of intensities
        sPhistsOfImg = [sPhistsOfImg; sPhist/trapz(binCenters,sPhist)];  %append normalised hist.
    end
    sPhists = [sPhists; sPhistsOfImg];                        %append image's histograms
end
end


%------------------------------------------------------- CLUSTER HISTOGRAMS
function cluster_idxs = clusterHistograms(hists, numClusters)
%Cluster superpixel histograms by chi-squared
    numSP = size(hists,1);    
    %compute chi-squared distance for all superpixel histograms
    histDistMatrix = zeros(numSP,numSP);
    parfor i=1:numSP
       for j=1:numSP
           if i>j %only calculate values above the diagonal
                histDistMatrix(i,j) = distChiSq(hists(i,:), hists(j,:));
           end
       end
    end
    histDistMatrix = histDistMatrix+histDistMatrix';          %store the values below the diagonal
    cluster_idxs = kmeans(histDistMatrix,numClusters);        %cluster the chi-squared distance matrix
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

