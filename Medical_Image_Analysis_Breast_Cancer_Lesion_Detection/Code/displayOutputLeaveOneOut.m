function displayOutputLeaveOneOut( testImagesThres, testImagesSegUpdate, testImagesSeg, evaluationCoeffs, images, groundTruths, nImages,verbose, resultsPath,imageNames )

if ~exist ( resultsPath, 'dir' )
    mkdir( resultsPath );
end


fontSize=12;
for i=1:nImages
    if verbose

        figure();
        subplot(221);imshow( images{i},[] );
        title( 'Original', 'FontSize', fontSize );

        subplot(222); imshow( testImagesSeg{i},[] );
        title('Lesion Probabilities', 'FontSize',fontSize );
        plotBoundaries( groundTruths{i} );

        subplot(223);imshow( testImagesSegUpdate{i},[] );
        title( 'Updated Probabilities', 'FontSize', fontSize );
        plotBoundaries( groundTruths{i} );

        subplot(224);imshow( testImagesThres{i},[] );
        title( 'Lesion Segmentation', 'FontSize', fontSize )
        plotBoundaries( groundTruths{i} );

    end 
    Ithresh=testImagesThres{i};
    imwrite( Ithresh, [resultsPath filesep 's' imageNames(i).name] );
end

figure();boxplot( evaluationCoeffs(:,1:2) );
end

function plotBoundaries( groundTruthTest )

B = bwboundaries(groundTruthTest,'noholes');

hold on
for k = 1:length(B)
    boundary = B{k};
    plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 1.5);
end

hold off

end