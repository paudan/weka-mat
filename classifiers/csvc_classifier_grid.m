function [classifier, tstart, telapsed] = csvc_classifier_grid(trainOBJ)
    clear java
    javaaddpath(sprintf('%slib%sgridSearch.jar',script_dir, filesep));
    import weka.core.*;
    import weka.filters.*;
    import weka.classifiers.functions.*;
    import weka.classifiers.meta.*;
    import
    classifier = weka.classifiers.meta.GridSearch;
    clsvm = weka.classifiers.functions.LibSVM;
    clsvm.setSVMType(SelectedTag(LibSVM.SVMTYPE_C_SVC, LibSVM.TAGS_SVMTYPE));
    clsvm.setKernelType(SelectedTag(LibSVM.KERNELTYPE_RBF, LibSVM.TAGS_KERNELTYPE));
    classifier.setFilter(AllFilter);
    classifier.setClassifier(clsvm);
    classifier.setEvaluation(SelectedTag(GridSearch.EVALUATION_ACC, GridSearch.TAGS_EVALUATION));
    classifier.setXMin(1);
    classifier.setXMax(100);
    classifier.setXStep(10);
    classifier.setXExpression('I');
    classifier.setXProperty('classifier.c');
    classifier.setYMin(-5);
    classifier.setYMax(2);
    classifier.setYStep(1);
    classifier.setYExpression('pow(BASE,I)');
    classifier.setXProperty('classifier.g');
    classifier.setGridIsExtendable(true);
    tstart = tic;
    classifier.buildClassifier(trainOBJ);
    telapsed = toc(tstart);
end