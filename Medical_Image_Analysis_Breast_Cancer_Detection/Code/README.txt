scriptLeaveOneOut: 	computes the leave-one-out validation on the training dataset (with groundtruth available).
scriptOptimizerParams:	determines the optimum parameters for maximising performance in the leave-one-out validation.
scriptFullTraining: 	computes the classifier model using all of the training images (re-uses features calculated in scriptLeaveOneOut, if available).
scriptTestingChallenge:	computes the segmentation of the testing images using the classifier model calculated in scriptFullTraining.

