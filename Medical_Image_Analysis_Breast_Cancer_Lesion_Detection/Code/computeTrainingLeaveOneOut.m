function treeModels = computeTrainingLeaveOneOut( featuresCell, groupCell, nImages )

treeModels=cell( 1,nImages );
parfor testIdx =1:nImages

    maskTrain=(1:nImages)~=testIdx; %train with other nImages-1 images

    %images and ground thruth datasets for training
    training=cell2mat( featuresCell(maskTrain)' );
    group=cell2mat( groupCell(maskTrain)' );
    
    %train the classifier for testing that image
    treeModels{testIdx}=TreeBagger(80,training,double(group),'Method','regression','MinLeaf',5,'Fboot',1);

end
