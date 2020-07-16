
clear all

datanumber=1000;  %% the number of data samples of each user

%Run DownloadCIFAR10 function to download CIFAR-10 dataset
%Run
% %% Prepare the CIFAR-10 dataset
% if ~exist('cifar10Train','dir')
%     disp('Saving the Images in folders. This might take some time...');    
%     saveCIFAR10AsFolderOfImages('cifar-10-batches-mat', pwd, true);
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%% data processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
categories = {'Deer','Dog','Frog','Cat','Bird','Automobile','Horse','Ship','Truck','Airplane'};

rootFolder = 'cifar10Test';
imds_test = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');


 categories = {'Deer','Dog','Frog','Cat','Bird','Automobile','Horse','Ship','Truck','Airplane'};

rootFolder = 'cifar10Train';
imds = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');
 
%%%%%%%%%%%%%%%%%%%%% IID dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
[imds1,imds2,imds3,imds4,imds5,imds6,imds7,imds8,imds9,imds10] = splitEachLabel(imds, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%% Non IID dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[imds1,imds2,imds3,imds4,imds5,imds6,imds7,imds8,imds9,imds10] = GetUnbalancedCIFAR(rootFolder, ratio)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% imds1 is the dataset of user 1. 


numberofneuron=50; % Number of neurons that consists of local FL model of each user
averagenumber=1;  % Average number of runing simulations. 
iteration=40;     % Total number of global FL iterations.
learningspeed=0.1; % Learning speed of each user
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%% coding setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
varSize = 32; 
v_fQRate = [1, 2];
v_nQuantizaers   = [...          % Curves
    0 ...                   % Dithered 3-D lattice quantization 
    1 ...                   % Dithered 2-D lattice quantization    
    1 ...                   % Dithered scalar quantization      
    1 ...                   % QSGD 
    1 ...                   % Uniform quantization with random unitary rotation    
    1 ...                   % Subsampling with 3 bits quantizers
    ];

global gm_fGenMat2D;
global gm_fLattice2D;
% Clear lattices
gm_fGenMat2D = [];
gm_fLattice2D = [];
% Do full search over the lattice
stSettings.OptSearch = 1;

stSettings.type =2;
stSettings.scale=1;
s_fRate=4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

localiterations=1;  % Number of local updates at each iteration.


finalerror=[];
averageerror=[];
kk=0;
proposed=0;



%%%%%%%%%%%%%%%%%%%%%%%% Matrix size of local FL model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w1length=5*5*3*32;

w2length=5*5*32*32;

w3length=5*5*32*64;

w4length=64*576;

w5length=10*64;

b1length=32;

b2length=32;

b3length=64;

b4length=64;

b5length=10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

error=zeros(iteration,1);

for userno=10:3:10    % Number of users.
    kk=kk+1;
    usernumber=userno; 
    
    
for average=1:1:averagenumber

    
    
wupdate=zeros(iteration,usernumber);   % local model for each user

%%%%%%%%%%%%% local model of each user%%%%%%%%%%%%%%%%%%%%%%%  
w1=[];
w2=[];
w3=[];
w4=[];
w5=[];
b1=[];
b2=[];
b3=[];
b4=[];
b5=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

wnew=zeros(5,5,3,32,usernumber);
lwnew=zeros(5,5,32,32,usernumber);
bnew=zeros(5,5,32,64,usernumber);
obnew=zeros(64,576,usernumber);
fwnew=zeros(10,64,usernumber);



%%%%%%%%%%%%% gradient of local FL models %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
deviationw=[];
deviationlw=[];
deviationb=[];
deviationob=[];
deviationofw=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%Building local FL model of each user  %%%%%%%%%%%%%%%%%%%%%%%%%
varSize = 32; 
layer = [
    imageInputLayer([varSize varSize 3]);
    convolution2dLayer(5,varSize,'Padding',2,'BiasLearnRateFactor',2); 
    maxPooling2dLayer(3,'Stride',2);
    reluLayer();
    convolution2dLayer(5,32,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);
    convolution2dLayer(5,64,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);
    fullyConnectedLayer(64,'BiasLearnRateFactor',2); 
    reluLayer();
    fullyConnectedLayer(length(categories),'BiasLearnRateFactor',2);
    softmaxLayer()
    classificationLayer()];

option = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.008, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 60, ...
    'Verbose', false);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




for i=1:1:iteration
    
%%%%%%%%%%%%%%%%%%%%%%%Setting of local FL model %%%%%%%%%%%%%%%%%%%%%%%%%%   
    if i==16
        option = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 60, ...
    'Verbose', false);
    
     elseif i==25
        option = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.002, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 60, ...
    'Verbose', false);
     elseif i==33
        option = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 60, ...
    'Verbose', false);
     elseif i==39
        option = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 60, ...
    'Verbose', false);
         elseif i==42
        option = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.00005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 60, ...
    'Verbose', false);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for user=1:1:usernumber
       
    
           clear netvaluable;
    Winstr1=strcat('net',int2str(user));     
     midstr=strcat('imds',int2str(user)); 
     
    eval(['imdss','=',midstr,';']);
    
if i > 1
   % Let global FL model to be the local FL model of each user, which is
   % equal to that the BS transmits the global FL model to the users  

      layer(2).Weights=globalw1;

    layer(5).Weights=globalw2;

     layer(8).Weights=globalw3;
     layer(11).Weights=globalw4;
    layer(13).Weights=globalw5;   
     
         layer(2).Bias=globalb1;

    layer(5).Bias=globalb2;

     layer(8).Bias=globalb3;
     layer(11).Bias=globalb4;
    layer(13).Bias=globalb5;   
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
       
      




[netvaluable, info] = trainNetwork(imdss, layer, option); % Train local FL model.


%%%%%%%%%%%%%%%%%%%calculate identification accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 labels = classify(netvaluable, imds_test);

% This could take a while if you are not using a GPU
confMat = confusionmat(imds_test.Labels, labels);
confMat = confMat./sum(confMat,2);
error(i,1)=mean(diag(confMat))+error(i,1); % Here, error is identification accuracy.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%% global model for each user, which consists of 4 matrices  

if i==1    
    globalw1=zeros(5,5,3,32);
globalw2=zeros(5,5,32,32);
globalw3=zeros(5,5,32,64);
globalw4=zeros(64,576);
globalw5=zeros(10,64);

    globalb1=zeros(1,1,32);
globalb2=zeros(1,1,32);
globalb3=zeros(1,1,64);
globalb4=zeros(64,1);
globalb5=zeros(10,1);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Record trained local FL model.

w1(:,:,:,:,user)=netvaluable.Layers(2).Weights;

w2(:,:,:,:,user)=netvaluable.Layers(5).Weights;

     w3(:,:,:,:,user)=netvaluable.Layers(8).Weights;
    w4(:,:,user)=netvaluable.Layers(11).Weights;
w5(:,:,user)=netvaluable.Layers(13).Weights;
     
     
b1(:,:,:,user)=netvaluable.Layers(2).Bias;

b2(:,:,:,user)=netvaluable.Layers(5).Bias;

   b3(:,:,:,user)=netvaluable.Layers(8).Bias;
   b4(:,:,user)=netvaluable.Layers(11).Bias;
   b5(:,:,user)=netvaluable.Layers(13).Bias;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     
if proposed==1
    
    
%%%%%%%%%%%%% Calculate the gradient of local FL model of each user%%%%%%%        
if i==1    
     
deviationw1= w1(:,:,:,:,user);
deviationw2=w2(:,:,:,:,user);
deviationw3= w3(:,:,:,:,user);
deviationw4=w4(:,:,user);
deviationw5=w5(:,:,user);

deviationb1=b1(:,:,:,user);
deviationb2=b2(:,:,:,user);
deviationb3=b3(:,:,:,user);
deviationb4=b4(:,:,user);
deviationb5= b5(:,:,user);

else
    
    
deviationw1= w1(:,:,:,:,user)-globalw1;
deviationw2=w2(:,:,:,:,user)-globalw2;
deviationw3= w3(:,:,:,:,user)-globalw3;
deviationw4=w4(:,:,user)-globalw4;
deviationw5=w5(:,:,user)-globalw5;

deviationb1=b1(:,:,:,user)-globalb1;
deviationb2=b2(:,:,:,user)-globalb2;
deviationb3=b3(:,:,:,user)-globalb3;
deviationb4=b4(:,:,user)-globalb4;
deviationb5= b5(:,:,user)-globalb5;    
        
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%% reshape the gradient of local FL model of each user%%%%%%%        

w1vector=reshape(deviationw1,[w1length,1]);

w2vector=reshape(deviationw2,[w2length,1]);

w3vector=reshape(deviationw3,[w3length,1]);

w4vector=reshape(deviationw4,[w4length,1]);

w5vector=reshape(deviationw5,[w5length,1]);   


b1vector=reshape(deviationb1,[b1length,1]);

b2vector=reshape(deviationb2,[b2length,1]);

b3vector=reshape(deviationb3,[b3length,1]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





    m_fH1 = [w1vector;w2vector;w3vector;w4vector;w5vector;...
             b1vector;b2vector;b3vector;deviationb4;deviationb5]; 
%    
   [m_fHhat1, ~] = m_fQuantizeData(m_fH1, s_fRate, stSettings); % coding and decoding
 
   bstart=w1length+w2length+w3length+w4length+w5length;
   
 %%%%%%%%%%%%%%%% reshape the gradient of the loss function after coding %%%%%%%%%%%%  
 deviationw1=reshape(m_fHhat1(1:w1length),[5,5,3,32]);
  deviationw2=reshape(m_fHhat1(w1length+1:w1length+w2length),[5,5,32,32]);
  deviationw3=reshape(m_fHhat1(w1length+w2length+1:w1length+w2length+w3length),[5,5,32,64]);
deviationw4=reshape(m_fHhat1(w1length+w2length+w3length+1:w1length+w2length+w3length+w4length),[64,576]);
deviationw5=reshape(m_fHhat1(w1length+w2length+w3length+w4length+1:bstart),[10,64]);

 deviationb1(1,1,:)=reshape(m_fHhat1(bstart+1:bstart+b1length),[1,1,32]);
  deviationb2(1,1,:)=reshape(m_fHhat1(bstart+b1length+1:bstart+b1length+b2length),[1,1,32]);
  deviationb3(1,1,:)=reshape(m_fHhat1(bstart+b1length+b2length+1:bstart+b1length+b2length+b3length),[1,1,64]);
deviationw4(:,1)=m_fHhat1(bstart+b1length+b2length+b3length+1:bstart+b1length+b2length+b3length+b4length);
deviationw5(:,1)=m_fHhat1(bstart+b1length+b2length+b3length+b4length+1:bstart+b1length+b2length+b3length+b4length+b5length);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 %%%%%%%%%%%%%%%% calculate the local FL model of each user after coding %%%%%%%%%%%%  

    if i==1
   
   w1(:,:,:,:,user)=deviationw1;
w2(:,:,:,:,user)=deviationw2;
 w3(:,:,:,:,user)=deviationw3;
w4(:,:,user)=deviationw4;
w5(:,:,user)=deviationw5;

b1(:,:,:,user)=deviationb1;
b2(:,:,:,user)=deviationb2;
b3(:,:,:,user)=deviationb3;
b4(:,:,user)=deviationb4;
b5(:,:,user)=deviationb5;
              
    else       
      w1(:,:,:,:,user)=deviationw1+globalw1;
w2(:,:,:,:,user)=deviationw2+globalw2;
 w3(:,:,:,:,user)=deviationw3+globalw3;
w4(:,:,user)=deviationw4+globalw4;
w5(:,:,user)=deviationw5+globalw5;

b1(:,:,:,user)=deviationb1+globalb1;
b2(:,:,:,user)=deviationb2+globalb2;
b3(:,:,:,user)=deviationb3+globalb3;
b4(:,:,user)=deviationb4+globalb4;
b5(:,:,user)=deviationb5+globalb5;     
        
    end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

end



 %%%%%%%%%%%%%%%% update the global FL model  %%%%%%%%%%%%  
globalw1=1/usernumber*sum(w1,5);  % global training model
globalw2=1/usernumber*sum(w2,5);  % global training model
globalw3=1/usernumber*sum(w3,5);
globalw4=1/usernumber*sum(w4,3);

globalw5=1/usernumber*sum(w5,3);

globalb1=1/usernumber*sum(b1,4);  % global training model
globalb2=1/usernumber*sum(b2,4);  % global training model
globalb3=1/usernumber*sum(b3,4);
globalb4=1/usernumber*sum(b4,3);

globalb5=1/usernumber*sum(b5,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%% without coding and encoding 
% 
% wglobal(i,:)=1/usernumber*sum(w,2);  % global training model
% lwglobal(i,:)=1/usernumber*sum(lw,2);  % global training model
% bglobal(i,:)=1/usernumber*sum(b,2);
% obglobal(i,:)=1/usernumber*sum(ob,2);


%tmp_net = netvaluable.saveobj;

% netvaluable.Layers(2).Weights =globalw1;
% tmp_net.Layers(5).Weights =globalw2;
% tmp_net.Layers(8).Weights =globalw3;
% tmp_net.Layers(11).Weights =globalw4;
% tmp_net.Layers(13).Weights =globalw5;
% 
% tmp_net.Layers(2).Bias =globalb1;
% tmp_net.Layers(5).Bias =globalb2;
% tmp_net.Layers(8).Bias =globalb3;
% tmp_net.Layers(11).Bias =globalb4;
% tmp_net.Layers(13).Bias =globalb5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


error(i,1)=error(i,1)/10; %%%% calculate the final error
end





    






end

end