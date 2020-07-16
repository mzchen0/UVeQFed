function saveCIFAR10AsFolderOfImages(inputPath, outputPath, varargin)
% saveCIFAR10AsFolderOfImages   Save the CIFAR-10 dataset as a folder of images
%   saveCIFAR10AsFolderOfImages(inputPath, outputPath) takes the CIFAR-10
%   dataset located at inputPath and saves it as a folder of images to the
%   directory outputPath. If inputPath or outputPath is an empty string, it
%   is assumed that the current folder should be used.
%
%   saveCIFAR10AsFolderOfImages(..., labelDirectories) will save the
%   CIFAR-10 data so that instances with the same label will be saved to
%   sub-directories with the name of that label.

% Check input directories are valid
if(~isempty(inputPath))
    assert(exist(inputPath,'dir') == 7);
end
if(~isempty(outputPath))
    assert(exist(outputPath,'dir') == 7);
end

% Check if we want to save each set with the same labels to its own
% directory.
if(isempty(varargin))
    labelDirectories = false;
else
    assert(nargin == 3);
    labelDirectories = varargin{1};
end

% Set names for directories
trainDirectoryName = 'cifar10Train';
testDirectoryName = 'cifar10Test';

% Create directories for the output
mkdir(fullfile(outputPath, trainDirectoryName));
mkdir(fullfile(outputPath, testDirectoryName));

if(labelDirectories)
    labelNames = {'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'};
    iMakeTheseDirectories(fullfile(outputPath, trainDirectoryName), labelNames);
    iMakeTheseDirectories(fullfile(outputPath, testDirectoryName), labelNames);
    for i = 1:5
        iLoadBatchAndWriteAsImagesToLabelFolders(fullfile(inputPath,['data_batch_' num2str(i) '.mat']), fullfile(outputPath, trainDirectoryName), labelNames, (i-1)*10000);
    end
    iLoadBatchAndWriteAsImagesToLabelFolders(fullfile(inputPath,'test_batch.mat'), fullfile(outputPath, testDirectoryName), labelNames, 0);
else
    for i = 1:5
        iLoadBatchAndWriteAsImages(fullfile(inputPath,['data_batch_' num2str(i) '.mat']), fullfile(outputPath, trainDirectoryName), (i-1)*10000);
    end
    iLoadBatchAndWriteAsImages(fullfile(inputPath,'test_batch.mat'), fullfile(outputPath, testDirectoryName), 0);
end
end

function iLoadBatchAndWriteAsImagesToLabelFolders(fullInputBatchPath, fullOutputDirectoryPath, labelNames, nameIndexOffset)
load(fullInputBatchPath);
data = data'; %#ok<NODEF>
data = reshape(data, 32,32,3,[]);
data = permute(data, [2 1 3 4]);
for i = 1:size(data,4)
    imwrite(data(:,:,:,i), fullfile(fullOutputDirectoryPath, labelNames{labels(i)+1}, ['image' num2str(i + nameIndexOffset) '.png']));
end
end

function iLoadBatchAndWriteAsImages(fullInputBatchPath, fullOutputDirectoryPath, nameIndexOffset)
load(fullInputBatchPath);
data = data'; %#ok<NODEF>
data = reshape(data, 32,32,3,[]);
data = permute(data, [2 1 3 4]);
for i = 1:size(data,4)
    imwrite(data(:,:,:,i), fullfile(fullOutputDirectoryPath, ['image' num2str(i + nameIndexOffset) '.png']));
end
end

function iMakeTheseDirectories(outputPath, directoryNames)
for i = 1:numel(directoryNames)
    mkdir(fullfile(outputPath, directoryNames{i}));
end
end