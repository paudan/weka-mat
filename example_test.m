clOBJ = WekaClassifier;
wekaOBJ = clOBJ.loadARFF('data/iris.arff');
wekaOBJ = clOBJ.apply_feature_selection(wekaOBJ, []);
[trainOBJ, testOBJ] = clOBJ.split_weka_dataset(wekaOBJ, 70);
classifier = clOBJ.create_random_forest_classifier(trainOBJ);
[accuracy, fmeasure] = clOBJ.perform_testing(testOBJ, classifier)

% clOBJ = WekaClassifier;
wekaOBJ = clOBJ.loadARFF('data/credit-g.arff');
[trainOBJ, testOBJ] = clOBJ.split_weka_dataset(wekaOBJ, 70);
classifier = nusvm_classifier(trainOBJ);
[accuracy, fmeasure] = clOBJ.perform_testing(testOBJ, classifier)