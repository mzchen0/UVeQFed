% Running this file will download CIFAR10 and place the images into a
% training folder and test folder in the current directory
% These will be used for the three demos in this folder. 
% Please note this will take a few minutes to run, but only needs to be run
% once.

% Copyright 2017 The MathWorks, Inc.

%% Download the CIFAR-10 dataset
if ~exist('cifar-10-batches-mat','dir')
    cifar10Dataset = 'cifar-10-matlab';
    disp('Downloading 174MB CIFAR-10 dataset...');   
    websave([cifar10Dataset,'.tar.gz'],...
        ['https://www.cs.toronto.edu/~kriz/',cifar10Dataset,'.tar.gz']);
    gunzip([cifar10Dataset,'.tar.gz'])
    delete([cifar10Dataset,'.tar.gz'])
    untar([cifar10Dataset,'.tar'])
    delete([cifar10Dataset,'.tar'])
end    
   
%% Prepare the CIFAR-10 dataset
if ~exist('cifar10Train','dir')
    disp('Saving the Images in folders. This might take some time...');    
    saveCIFAR10AsFolderOfImages('cifar-10-batches-mat', pwd, true);
end

