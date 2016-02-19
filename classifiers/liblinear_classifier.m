function [classifier, tstart, telapsed] = liblinear_classifier(trainOBJ, type)
% Create LibLINEAR classifier, which is called from Weka environment
% Set type of solver (default: 1) for multi-class classification
%      0 -- L2-regularized logistic regression (primal)
%      1 -- L2-regularized L2-loss support vector classification (dual)
%      2 -- L2-regularized L2-loss support vector classification (primal)
%      3 -- L2-regularized L1-loss support vector classification (dual)
%      4 -- support vector classification by Crammer and Singer
%      5 -- L1-regularized L2-loss support vector classification
%      6 -- L1-regularized logistic regression
%      7 -- L2-regularized logistic regression (dual)
if type < 0 || type > 7
    error('LibLINEAR SVM type must be between 0 and 7')
end
javaaddpath(sprintf('%slib%sLibLINEAR.jar',script_dir, filesep));
javaaddpath(sprintf('%slib%sliblinear-1.51-with-deps.jar',script_dir, filesep));
javaaddpath(sprintf('%slib%sliblinear-1.92.jar',script_dir, filesep));
import weka.core.*;
import weka.classifiers.functions.*;
classifier = weka.classifiers.functions.LibLINEAR;
classifier.setSVMType(SelectedTag(type, LibLINEAR.TAGS_SVMTYPE));
classifier.setCost(5);
classifier.setBias(1.0); 
tstart = tic;
classifier.buildClassifier(trainOBJ);
telapsed = toc(tstart);
end