classdef WekaClassifier
    %WEKACLASSIFIER Creates and trains Weka classifier from MATLAB/Octave environment
    
    properties(Access = private)
        weka_st = true;
    end
    
    methods(Access = private)
        function display_testing_results(this, classifier, eval)
            fprintf(this.classifier_string(classifier))
            fprintf('=== Classifier model ===\n\n')
            disp( char(classifier.toString()) )
            fprintf('=== Summary ===\n')
            disp( char(eval.toSummaryString()) )
            disp( char(eval.toClassDetailsString()) )
            disp( char(eval.toMatrixString()) )
        end

        function classifierstr = classifier_string(this, classifier)
            if ~this.weka_st
                error('Weka is not initialized. See "help wekaPathCheck" for more details')
            end
            classifierstr = sprintf('%s %s', char(classifier.getClass().getSimpleName()), ...
                char(weka.core.Utils.joinOptions(classifier.getOptions())));
        end

    end
    
    methods
        
        function this = WekaClassifier
            addpath(sprintf('%sclassifiers%s',script_dir, filesep));
            addpath(sprintf('%sdatabase%s',script_dir, filesep));
            addpath(sprintf('%sdata%s',script_dir, filesep)); 
            addpath(sprintf('%slib%s',script_dir, filesep)); 
            addpath(sprintf('%smatlab2weka%s',script_dir, filesep));
            javaaddpath(sprintf('%slib%sweka.jar',script_dir, filesep));
            javaaddpath(sprintf('%slib%sSMOTE.jar',script_dir, filesep));
            javaaddpath(sprintf('%slib%sLibSVM.jar',script_dir, filesep));
            javaaddpath(sprintf('%slib%slibsvm-3.12.jar',script_dir, filesep));
            if ~wekaPathCheck
                this.weka_st = false;
            end
        end
        
        function wekaOBJ = create_weka_dataset(this, relname, data, labels, attributes)
            if ~this.weka_st
                error('Weka is not initialized. See "help wekaPathCheck" for more details')
            end
            wekaOBJ = matlab2weka(relname, [attributes; 'Class'], [data labels], size(data, 2)+1);
            import weka.filters.unsupervised.attribute.*;
            import weka.filters.*;
            rpfilter = NumericToNominal;
            rpfilter.setAttributeIndicesArray([size(data, 2)]);
            rpfilter.setInputFormat(wekaOBJ);
            wekaOBJ = Filter.useFilter(wekaOBJ, rpfilter);
            wekaOBJ.setClassIndex(size(data, 2));
        end
        
        function [trainOBJ, testOBJ] = split_weka_dataset(this, wekaOBJ, trainPerc)
            if ~this.weka_st
                error('Weka is not initialized. See "help wekaPathCheck" for more details')
            end
            import weka.filters.*;
            import weka.filters.unsupervised.attribute.*;
            import weka.filters.unsupervised.instance.*;
            rpfilter = RemovePercentage;
            rpfilter.setPercentage(trainPerc);
            rpfilter.setInvertSelection(true);
            rpfilter.setInputFormat(wekaOBJ);
            trainOBJ = Filter.useFilter(wekaOBJ, rpfilter);
            rpfilter = RemovePercentage;
            rpfilter.setPercentage(trainPerc);
            rpfilter.setInvertSelection(false);
            rpfilter.setInputFormat(wekaOBJ);
            testOBJ = Filter.useFilter(wekaOBJ, rpfilter);
            normfilter = Normalize;
            normfilter.setInputFormat(trainOBJ);
            normfilter.setIgnoreClass(true);
            trainOBJ = Filter.useFilter(trainOBJ, normfilter);
            testOBJ = Filter.useFilter(testOBJ, normfilter);
            trainOBJ.setClassIndex(trainOBJ.numAttributes()-1);
            testOBJ.setClassIndex(testOBJ.numAttributes()-1);
        end
        
        function [accuracy, fmeasure, predictedClass, classProbs] = perform_testing(this, testOBJ, classifier)
            if ~this.weka_st
                error('Weka is not initialized. See "help wekaPathCheck" for more details')
            end
            ev = weka.classifiers.Evaluation(testOBJ);
            if ~isempty(classifier)
                try
                    ev.evaluateModel(classifier, testOBJ, javaArray('java.lang.Object',1));  
                    this.display_testing_results(classifier, ev); 
                    accuracy = ev.pctCorrect;
                    fmeasure = ev.weightedFMeasure;
                    for t=0:testOBJ.numInstances -1  
                       classProbs(t+1,:) = (classifier.distributionForInstance(testOBJ.instance(t)))';
                    end
                    [~,predictedClass] = max(classProbs,[],2);
                    predictedClass = predictedClass - 1;
                catch e
                    e.message
                end
            end
        end
        
        function dataTrain = scaleData(~, dataTrain, minimums, ranges)
            dataTrain = (dataTrain - repmat(minimums,size(dataTrain,1),1)) * ...
                    spdiags(1./ranges',0,size(dataTrain,2),size(dataTrain,2)); 
        end
        
        function [classifier, tstart, telapsed] = create_random_forest_classifier(this, trainOBJ)
        % Example of Weka classifier building
            if ~this.weka_st
                error('Weka is not initialized. See "help wekaPathCheck" for more details')
            end
            import weka.classifiers.trees.*;
            classifier = weka.classifiers.trees.RandomForest;
            classifier.setNumTrees(100);
            tstart = tic;
            classifier.buildClassifier(trainOBJ); 
            telapsed = toc(tstart);
        end
        
        function [trainOBJ] = apply_smote(this, trainOBJ)
        % Apply SMOTE filter for imbalanced learning
            if ~this.weka_st
                error('Weka is not initialized. See "help wekaPathCheck" for more details')
            end
            import weka.filters.*;
            import weka.filters.supervised.instance.*;
            try 
                sampler = weka.filters.supervised.instance.SMOTE;
                sampler.setInputFormat(trainOBJ);
                sampler.setNearestNeighbors(5);
                trainOBJ = weka.filters.Filter.useFilter(trainOBJ, sampler);
            catch 
            end
        end
        
        function [wekaOBJ, attr_selected] = apply_feature_selection(this, wekaOBJ, attribs)
        % wekaOBJ is Java object created using create_weka_dataset
            if ~this.weka_st
                error('Weka is not initialized. See "help wekaPathCheck" for more details')
            end
            import weka.attributeSelection.*;
            attrsel = weka.attributeSelection.AttributeSelection; 
            ev = weka.attributeSelection.CfsSubsetEval;
            search = weka.attributeSelection.GreedyStepwise;
            attrsel.setEvaluator(ev);
            attrsel.setSearch(search);
            attrsel.SelectAttributes(wekaOBJ);
            sel_indices = attrsel.selectedAttributes;
            sel_indices = sel_indices(1:end-1)+1;
            if ~isempty(attribs)
                attr_selected = attribs(sel_indices);
            end
            wekaOBJ = attrsel.reduceDimensionality(wekaOBJ);
            wekaOBJ.setClassIndex(wekaOBJ.numAttributes-1);
        end
        
        function [wekaOBJ] = loadARFF(~, filename)
            wekaOBJ = loadARFF(filename);
        end

    end
end

