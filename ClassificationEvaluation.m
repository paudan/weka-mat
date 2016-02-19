classdef ClassificationEvaluation
    % Classification performance extracted from confusion matrix 

    properties(SetAccess = private)
      confusionMatrix
      labels
      predictions
    end
 
    methods(Access = private)
         function weightedRatio = weightedRatio(evalobj, ratiolist)
            classCounts = zeros(length(evalobj.confusionMatrix), 1);
            ratioTotal = 0;
            for i=1:length(evalobj.confusionMatrix)
                classCounts(i) = nansum(evalobj.confusionMatrix(i, :));
                ratioTotal = ratioTotal + ratiolist(i)*classCounts(i);
            end
            weightedRatio = ratioTotal/sum(classCounts);
        end
    end
    
    methods 
        function evalobj = ClassificationEvaluation(conf_matrix, labels, predictions)
            evalobj.confusionMatrix = conf_matrix;
            evalobj.labels = labels;
            evalobj.predictions = predictions;
        end
        
        function recall = recall(evalobj, index)      
            recall = evalobj.truePositiveRate(index);
        end

        function recalls = recallValues(evalobj)
            recalls = evalobj.truePositiveRates;
        end
        
        function precision = precision(evalobj, index)
            correct = 0; total = 0;
            for i = 1:size(evalobj.confusionMatrix, 1)
                if (i == index)
                    correct = correct + evalobj.confusionMatrix(index, i);
                end
                total = total + evalobj.confusionMatrix(index, i);
            end
            precision = correct/total;
        end
        
        function prec = precisionValues(evalobj)
            prec = zeros(length(evalobj.confusionMatrix), 1);
            for i = 1:size(evalobj.confusionMatrix, 1)
                prec(i) = evalobj.precision(i);
            end
         end

        function fnr = falseNegativeRate(evalobj, index)
            len = size(evalobj.confusionMatrix, 1);
            incorrect = 0; total = 0;
            for i = 1:len
                if (i == index)
                    for j = 1:len
                        if (j ~= index)
                            incorrect = incorrect + evalobj.confusionMatrix(i, j);
                        end
                        total = total + evalobj.confusionMatrix(i, j);
                    end
                end
            end
            fnr = incorrect / total;
        end
        
        function fnr = falseNegativeRates(evalobj)
            fnr = zeros(length(evalobj.confusionMatrix), 1);
            for i = 1:size(evalobj.confusionMatrix, 1)
                fnr(i) = evalobj.falseNegativeRate(i);
            end
         end

        function tpr = truePositiveRate(evalobj, index)
            correct = 0; total = 0;
            for j = 1:size(evalobj.confusionMatrix, 1)
                if j == index
                    correct = correct + evalobj.confusionMatrix(index, j);
                end
                total = total + evalobj.confusionMatrix(index, j);
            end
            tpr = correct / total;
        end
        
        function tpr = truePositiveRates(evalobj)
            tpr = zeros(length(evalobj.confusionMatrix), 1);
            for i = 1:size(evalobj.confusionMatrix, 1)
                tpr(i) = evalobj.truePositiveRate(i);
            end
        end
         
        function fpr = falsePositiveRate(evalobj, index)
            len = size(evalobj.confusionMatrix, 1);
            incorrect = 0; total = 0;
            for i = 1:len
                if (i ~= index)
                    for j = 1:len
                        if (j == index)
                            incorrect = incorrect + evalobj.confusionMatrix(i, j);
                        end
                        total = total + evalobj.confusionMatrix(i, j);
                    end
                end
            end
            fpr = incorrect / total;
        end
        
        function fpr = falsePositiveRates(evalobj)
            fpr = zeros(length(evalobj.confusionMatrix), 1);
            for i = 1:size(evalobj.confusionMatrix, 1)
                fpr(i) = evalobj.falsePositiveRate(i);
            end
        end
        
        function tnr = trueNegativeRate(evalobj, index)
            len = size(evalobj.confusionMatrix, 1);
            correct = 0; total = 0;
            for i = 1:len
                if (i ~= index)
                    for j = 1:len
                        if (j ~= index)
                            correct = correct + evalobj.confusionMatrix(i, j);
                        end
                        total = total + evalobj.confusionMatrix(i, j);
                    end
                end
            end
            tnr = correct / total;  
        end

        function tnr = trueNegativeRates(evalobj)
            tnr = zeros(length(evalobj.confusionMatrix), 1);
            for i = 1:size(evalobj.confusionMatrix, 1)
                tnr(i) = evalobj.trueNegativeRate(i);
            end
        end
  
        function ntp = numTruePositives(evalobj, index)
            for j = 1:size(evalobj.confusionMatrix, 1)
                if j ~= index
                    ntp = ntp + evalobj.confusionMatrix(index, j);
                end
            end
        end
        
        function ntn = numTrueNegatives(evalobj, index)
            for i = 1:len
                if (i ~= index)
                    for j = 1:len
                        if (j ~= index)
                            ntn = ntn + evalobj.confusionMatrix(i, j);
                        end
                    end
                end
            end
        end
        
        function nfp = numFalsePositives(evalobj, index)
            for i = 1:len
                if (i ~= index)
                    for j = 1:len
                        if (j == index)
                            nfp = nfp + evalobj.confusionMatrix(i, j);
                        end
                    end
                end
            end
        end
        
        function nfn = numFalseNegatives(evalobj,index)
            for i = 1:len
                if (i == index)
                    for j = 1:len
                        if (j ~= index)
                            nfn = nfn + evalobj.confusionMatrix(i, j);
                        end
                    end
                end
            end
        end
        
        function fmeas = fMeasure(evalobj, index)
            precision = evalobj.precision(index);
            recall = evalobj.recall(index);
            fmeas = 2 * precision * recall / (precision + recall);
        end
        
        function prec = fMeasureValues(evalobj)
            prec = zeros(length(evalobj.confusionMatrix), 1);
            for i = 1:size(evalobj.confusionMatrix, 1)
                prec(i) = evalobj.fMeasure(i);
            end
        end
         
        function acc = pctCorrect(evalobj)
            acc = sum(diag(evalobj.confusionMatrix))/sum(evalobj.confusionMatrix(:));
        end
        
        function acc = pctIncorrect(evalobj)
            acc = 1 - evalobj.pctCorrect;
        end
        
        function auc = areaUnderCurve(evalobj, class)
            t = tabulate(evalobj.labels);
            if t(class, 2) > 0
                [~,~,~,auc] = perfcurve(evalobj.labels, evalobj.predictions, num2str(class));
            else
                auc = NaN;
            end
        end
        
        function auc = AUCValues(evalobj)
            auc = zeros(length(evalobj.confusionMatrix), 1);
            t = tabulate(evalobj.labels);
            for i = t(:, 1)'
                auc(i) = evalobj.areaUnderCurve(i);
            end
        end
        
        function weightedTPR = weightedTruePositiveRate(evalobj)
            weightedTPR = evalobj.weightedRatio(evalobj.truePositiveRates);
        end
        
        function weightedTNR = weightedTrueNegativeRate(evalobj)
            weightedTNR = evalobj.weightedRatio(evalobj.trueNegativeRates);
        end
        
        function weightedFPR = weightedFalsePositiveRate(evalobj)
            weightedFPR = evalobj.weightedRatio(evalobj.falsePositiveRates);
        end   
        
        function weightedFNR = weightedFalseNegativeRate(evalobj)
            weightedFNR = evalobj.weightedRatio(evalobj.falseNegativeRates);
        end    
        
        function weightedPrec = weightedPrecision(evalobj)
            weightedPrec = evalobj.weightedRatio(evalobj.precisionValues);
        end
        
        function weightedFMeas = weightedFMeasure(evalobj)
            weightedFMeas = evalobj.weightedRatio(evalobj.fMeasureValues);
        end
        
        function weightedAUC = weightedAUC(evalobj)
            classCounts = zeros(length(evalobj.confusionMatrix), 1);
            ratioTotal = 0;
            for i=1:size(evalobj.confusionMatrix, 1)
                classCounts(i) = nansum(evalobj.confusionMatrix(i, :));
                auc = evalobj.areaUnderCurve(i);
                if isnan(auc)
                    auc = 0;
                end
                ratioTotal = ratioTotal + auc*classCounts(i);
            end
           weightedAUC = ratioTotal/sum(classCounts);
        end
        
        function numClasses = numClasses(evalobj)
            numClasses = size(evalobj.confusionMatrix, 1);
        end
    end
end
   
