function [testImagesThres testImagesSegUpdate testImagesSeg evaluationResults ] = computeTestingLeaveOneOut( featuresCell, sPlabels, treeModels, groundTruths, nImages )

testImagesSeg = cell( 1,nImages );          %output from classifier: probability of lesion
testImagesSegUpdate = cell( 1,nImages );    %output after post-processing
testImagesThres = cell( 1,nImages );        %output after thresholding
evaluationResults = zeros( nImages,4 );     %comparison with ground truth

for testIdx = 1:nImages
    test = featuresCell{testIdx};           %features of superpixels to be classified
    [outputClassifier] = predict(treeModels{testIdx},test); %predict probability of each superpixel as lesion, using the model trained without its image
    Iseg = outputClassifier(sPlabels{testIdx}); %copy superpixels' probabilities to all pixels within
    Iseg = Iseg-min(Iseg(:));               %make probability map have a min of 0
    Iseg = Iseg/max(Iseg(:));               %make probability map have a max of 1
    testImagesSeg{testIdx} = Iseg;                                        %store output of classifier
    
    %parameters from optimiser output
    params = [0.4086    0.4923    0.7228    0.0312    0.5234    0.1499    0.0569    0.2987];
    
    %apply post-processing of reducing the probability of superpixels far from [x_0,y_0]
    sigma_x=params(5);     sigma_y=params(6);      x_0=params(7);     y_0=params(8);
    Iseg_post = updateProbs_center(Iseg, sPlabels{testIdx}, sigma_x, sigma_y, x_0, y_0);
    testImagesSegUpdate{testIdx} = Iseg_post;                             %store post-processed output
    
    %threshold the image to determine the final segmentation
    a = params(1);      b = params(2);      c = params(3);      d = params(4);
    threshValue = min(max(Iseg_post(:))-a*std(outputClassifier),b)*c + d; %determine the relevant threshold
    testImagesThres{testIdx} = Iseg_post>threshValue;                     %threshold post-processed probability map and store final segmentation
    
    %evaluate the segmentation with respect to the ground truth
    [evaluationResults(testIdx,1) evaluationResults(testIdx,2) evaluationResults(testIdx,3)...
        evaluationResults(testIdx,4)] = sevaluate( groundTruths{testIdx}, testImagesThres{testIdx} );
    
end