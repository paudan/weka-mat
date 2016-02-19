function [classifier, tstart, telapsed] = csvc_classifier_selcv(trainOBJ, folds)
% C-SVC classifier which applies cross validation for parameter selection
    if nargin == 1 || isempty(folds)
        folds = 5;
    end
    import weka.core.*;
    import weka.classifiers.functions.*;
    import weka.classifiers.meta.*;
    classifier = CVParameterSelection;     
    clsvm = weka.classifiers.functions.LibSVM;
    clsvm.setSVMType(SelectedTag(LibSVM.SVMTYPE_C_SVC, LibSVM.TAGS_SVMTYPE));
    clsvm.setKernelType(SelectedTag(LibSVM.KERNELTYPE_RBF, LibSVM.TAGS_KERNELTYPE));
    classifier.setClassifier(clsvm);
    classifier.setNumFolds(folds);  % using 5-fold CV
    classifier.addCVParameter('C 1 100 10');
    classifier.addCVParameter('G 0.01 0.1 10');
    tstart = tic;
    classifier.buildClassifier(trainOBJ);
    telapsed = toc(tstart);
end

