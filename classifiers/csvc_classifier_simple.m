function [classifier, tstart, telapsed] = csvc_classifier_simple(trainOBJ, C, gamma)
    if nargin == 1 || isempty(C)
        C = 100;
        gamma=0.01;
    end
    if nargin == 2 || isempty(gamma)
        gamma=0.01;
    end
    import weka.core.*;
    import weka.classifiers.functions.*;
    classifier = weka.classifiers.functions.LibSVM;
    classifier.setSVMType(SelectedTag(LibSVM.SVMTYPE_C_SVC, LibSVM.TAGS_SVMTYPE));
    classifier.setKernelType(SelectedTag(LibSVM.KERNELTYPE_RBF, LibSVM.TAGS_KERNELTYPE));
    classifier.setCost(C);
    classifier.setGamma(gamma);
    tstart = tic;
    classifier.buildClassifier(trainOBJ);
    telapsed = toc(tstart);
end