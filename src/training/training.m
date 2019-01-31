%%Interactive Transfer Learning Using DenseNet201
%%This program fine-tunes a pretrained DenseNet201 to classify a new collection of images.
%%Images (3 bands, 8bit, tif) are stored in 4 subfolders of Crevasse, Ice-Slush, LargeRiver, and SmallRiver in the Data folder 
clc
clear all
close all

%% Split the data into 70% training and 30% test data.
imds = imageDatastore('Data','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

%% Display some sample images.
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

%% Load a pretrained DenseNet201 network.
net = densenet201;

%% Display the network architecture.
inputSize = net.Layers(1).InputSize

%% Extract the layer graph from the trained network.
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

%% Find the names of the two layers to replace.
[learnableLayer,classLayer] = findLayersToReplace(lgraph); 

%% Replace the fully connected layer with a new fully connected layer with the number of outputs equal to the number of classes in the new data set.
numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

%% Replace the classification layer with a new one without class labels.
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%% Plot the new layer graph and zoom in on the last layers of the network.
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

%% Use an augmented image datastore to automatically resize the training images.
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

%% Automatically resize the validation images without performing further data augmentation.
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%% Specify the training options.
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Train the network using the training data.
net = trainNetwork(augimdsTrain,lgraph,options);

%% Classify the validation images using the fine-tuned network, and calculate the classification accuracy.
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

%% Display sample validation images with predicted labels and predicted probabilities.
idx = randperm(numel(imdsValidation.Files),25);
figure
for i = 1:25
    subplot(5,5,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + "," + num2str(100*max(probs(idx(i),:)),3) + "%");
end







