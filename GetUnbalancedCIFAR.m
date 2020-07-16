function [imds1,imds2,imds3,imds4,imds5,imds6,imds7,imds8,imds9,imds10] = GetUnbalancedCIFAR(rootFolder, ratio)

categories = {'Deer','Dog','Frog','Cat','Bird','Automobile','Horse','Ship','Truck','Airplane'};


Catimds = cell(10,1); 
Shareimds = cell(10,1);
Indimds = cell(10,1);
for kk=1:length(categories)
    % Read each label    
    Catimds{kk} =  imageDatastore(fullfile(rootFolder, categories(:,kk)), ...
                                    'LabelSource', 'foldernames');
    % Divide into shared and individual labels                     
    [Indimds{kk}, Shareimds{kk}] = splitEachLabel(Catimds{kk},  ratio);  
end 
% split joint labels
Jointimds = imageDatastore(cat(1,Shareimds{1}.Files,Shareimds{2}.Files,Shareimds{3}.Files,Shareimds{4}.Files ...
                                ,Shareimds{5}.Files,Shareimds{6}.Files,Shareimds{7}.Files,Shareimds{8}.Files ...
                                ,Shareimds{9}.Files,Shareimds{10}.Files));
Jointimds.Labels = cat(1,Shareimds{1}.Labels,Shareimds{2}.Labels,Shareimds{3}.Labels,Shareimds{4}.Labels ...
                         ,Shareimds{5}.Labels,Shareimds{6}.Labels,Shareimds{7}.Labels,Shareimds{8}.Labels ...
                         ,Shareimds{9}.Labels,Shareimds{10}.Labels);

[imds1A,imds2A,imds3A,imds4A,imds5A,imds6A,imds7A,imds8A,imds9A,imds10A] = splitEachLabel(Jointimds, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1);  

% Generate unbalanced data set
imds1 = imageDatastore(cat(1, Indimds{1}.Files, imds1A.Files));
imds1.Labels = cat(1, Indimds{1}.Labels, imds1A.Labels);

imds2 = imageDatastore(cat(1, Indimds{2}.Files, imds2A.Files));
imds2.Labels = cat(1, Indimds{2}.Labels, imds2A.Labels);

imds3 = imageDatastore(cat(1, Indimds{3}.Files, imds3A.Files));
imds3.Labels = cat(1, Indimds{3}.Labels, imds3A.Labels);

imds4 = imageDatastore(cat(1, Indimds{4}.Files, imds4A.Files));
imds4.Labels = cat(1, Indimds{4}.Labels, imds4A.Labels);

imds5 = imageDatastore(cat(1, Indimds{5}.Files, imds5A.Files));
imds5.Labels = cat(1, Indimds{5}.Labels, imds5A.Labels);

imds6 = imageDatastore(cat(1, Indimds{6}.Files, imds6A.Files));
imds6.Labels = cat(1, Indimds{6}.Labels, imds6A.Labels);

imds7 = imageDatastore(cat(1, Indimds{7}.Files, imds7A.Files));
imds7.Labels = cat(1, Indimds{7}.Labels, imds7A.Labels);

imds8 = imageDatastore(cat(1, Indimds{8}.Files, imds8A.Files));
imds8.Labels = cat(1, Indimds{8}.Labels, imds8A.Labels);

imds9 = imageDatastore(cat(1, Indimds{9}.Files, imds9A.Files));
imds9.Labels = cat(1, Indimds{9}.Labels, imds9A.Labels);

imds10 = imageDatastore(cat(1, Indimds{10}.Files, imds10A.Files));
imds10.Labels = cat(1, Indimds{10}.Labels, imds10A.Labels);