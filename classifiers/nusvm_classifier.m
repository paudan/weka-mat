function [classifier, tstart, telapsed] = nusvm_classifier(trainOBJ)
    import weka.core.*;
    import weka.classifiers.functions.*;
    nu = 0.9;
    classifier = weka.classifiers.functions.LibSVM;
    classifier.setSVMType(SelectedTag(LibSVM.SVMTYPE_NU_SVC, LibSVM.TAGS_SVMTYPE));
    classifier.setKernelType(SelectedTag(LibSVM.KERNELTYPE_RBF, LibSVM.TAGS_KERNELTYPE));
    classifier.setGamma(0.001);
    cl_built = false;
    while ~cl_built
        try
            tstart = tic;
            classifier.setNu(nu);
            classifier.buildClassifier(trainOBJ);
            telapsed = toc(tstart);
            cl_built = true;
        catch 
            nu = nu - 0.1;
            if nu == 0
                cl_built = true;
            end
        end
    end
end