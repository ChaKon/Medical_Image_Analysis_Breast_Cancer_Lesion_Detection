function s = diceCoefObjectiveFunctionRandomForest(params)

global groundTruths
global testedImages
global nImages
global sPlabels
global deviations

%load paramater values
a=params(1);
b=params(2);
c=params(3);
d=params(4);
sigma_x =params(5);
sigma_y =params(6);
x_0 =params(7);
y_0 =params(8);
testImagesDice = zeros( 1,nImages );


parfor testIdx =1:nImages
    Iseg=testedImages{testIdx};
    
    Iseg=updateProbs_center( Iseg, sPlabels{testIdx},sigma_x,sigma_y,x_0,y_0 );
    threshValue=min( max(Iseg(:))-a*deviations(testIdx),b )*c + d;
    Ithresh=Iseg>threshValue;
    [~,testImagesDice(testIdx)]=sevaluate( logical(groundTruths{testIdx}), Ithresh );
    
end


s =-( mean( testImagesDice ) );% + median( testImagesDice ) );