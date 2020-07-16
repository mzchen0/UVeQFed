
clear all

datanumber=1000;  %% the number of data samples of each user

eachdatanumber=100; % The number of each digit




%%%%%%%%%%%%%%%%%%%%%data processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[trainingdata, traingnd] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
trainingdata = double(reshape(trainingdata, size(trainingdata,1)*size(trainingdata,2), []).');
traingnd = double(traingnd);
traingnd(traingnd==0)=10;


    
trainingdata1=trainingdata(find(traingnd==1),:);
trainingdata2=trainingdata(find(traingnd==2),:);
trainingdata3=trainingdata(find(traingnd==3),:);
trainingdata4=trainingdata(find(traingnd==4),:);
trainingdata5=trainingdata(find(traingnd==5),:);
trainingdata6=trainingdata(find(traingnd==6),:);
trainingdata7=trainingdata(find(traingnd==7),:);
trainingdata8=trainingdata(find(traingnd==8),:);
trainingdata9=trainingdata(find(traingnd==9),:);
trainingdata10=trainingdata(find(traingnd==10),:);




traingndold=traingnd;
traingnd=dummyvar(traingnd); 

traingnd1=traingnd(find(traingndold==1),:);
traingnd2=traingnd(find(traingndold==2),:);
traingnd3=traingnd(find(traingndold==3),:);
traingnd4=traingnd(find(traingndold==4),:);
traingnd5=traingnd(find(traingndold==5),:);
traingnd6=traingnd(find(traingndold==6),:);
traingnd7=traingnd(find(traingndold==7),:);
traingnd8=traingnd(find(traingndold==8),:);
traingnd9=traingnd(find(traingndold==9),:);
traingnd10=traingnd(find(traingndold==10),:);



[testdata, testgnd] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
testdata = double(reshape(testdata, size(testdata,1)*size(testdata,2), []).');
testgnd = double(testgnd);
testgnd(testgnd==0)=10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





numberofneuron=50; % Number of neurons that consists of local FL model of each user
averagenumber=1;% Average number of runing simulations. 
iteration=300; % Total number of global FL iterations.
learningspeed=0.1; % Learning speed of each user


%%%%%%%%%%%%%%%%%%%%%%%%%%coding setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

v_fQRate = [1, 2];
v_nQuantizaers   = [...          % Curves
    0 ...                   % Dithered 3-D lattice quantization 
    1 ...                   % Dithered 2-D lattice quantization    
    1 ...                   % Dithered scalar quantization      
    1 ...                   % QSGD 
    1 ...                   % Uniform quantization with random unitary rotation    
    1 ...                   % Subsampling with 3 bits quantizers
    ];

% -----------------------------------------------------------------------


global gm_fGenMat2D;
global gm_fLattice2D;
% Clear lattices
gm_fGenMat2D = [];
gm_fLattice2D = [];
% Do full search over the lattice
stSettings.OptSearch = 1;
stSettings.scale=2;

stSettings.type =4;
s_fRate=4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


localiterations=1; % Number of local updates at each iteration.


finalerror=[];
averageerror=[];
kk=0;
proposed=1;        % proposed=1 indicates the proposed algorithm 
                   % proposed=0 indicates the comparison algorithm 


for userno=15:3:15   % Number of users.
    kk=kk+1;
    usernumber=userno; 
    
    
for average=1:1:averagenumber

    
%%%%%%%%%%%%% local model for each user, which consists of 4 matrices  
wupdate=zeros(iteration,usernumber);   
w=[];
lw=[];
b=[];

ob=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

wnew=zeros(numberofneuron,usernumber);
lwnew=zeros(numberofneuron,usernumber);
bnew=zeros(numberofneuron,usernumber);

obnew=zeros(1,usernumber);

%%%%%%%%%%%%% global model for each user, which consists of 4 matrices  

wglobal=[];            % global training model
lwglobal=[];            % global training model
bglobal=[];  
obglobal=[]; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% gradient of local FL models %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
deviationw=[];
deviationlw=[];
deviationb=[];
deviationob=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%Building local FL model for user 1 %%%%%%%%%%%%%%%%%%%%%%%%%
net1 = patternnet(numberofneuron);
 %  net1.trainFcn = 'trainscg';
    %  net1.trainFcn = 'traingd';
     net1.inputs{1}.processFcns={};
 net1.outputs{2}.processFcns={};
%   net1.divideFcn = '';
net1.trainParam.epochs = localiterations;

net1.trainParam.showWindow = 0;
%net1.inputs{1}.size=500;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%net1.trainParam.lr=learningspeed;

net2 = patternnet(numberofneuron);
   
  % net2.trainFcn = 'trainscg';
     net2.inputs{1}.processFcns={};
 net2.outputs{2}.processFcns={};
 %net2.inputs{1}.size=500;

%  
%  net2.divideFcn = '';
net2.trainParam.epochs = localiterations;
net2.trainParam.showWindow = 0;
%net2.trainParam.lr=learningspeed;

net3 = patternnet(numberofneuron);
    net3.inputs{1}.processFcns={};
 net3.outputs{2}.processFcns={};
 net3.trainParam.showWindow = 0;
 %  net3.trainFcn = 'trainscg';
%net3.inputs{1}.size=500;

%  net3.divideFcn = '';
net3.trainParam.epochs = localiterations;


if usernumber>3
%net3.trainParam.lr=learningspeed;

net4 = patternnet(numberofneuron);
   
    net4.inputs{1}.processFcns={};
 net4.outputs{2}.processFcns={};

%  net4.divideFcn = ''; 
%  net4.trainFcn = 'traingd';

net4.trainParam.epochs = localiterations;
net4.trainParam.showWindow = 0;

%net4.trainParam.lr=learningspeed;


net5 = patternnet(numberofneuron);
    net5.inputs{1}.processFcns={};
 net5.outputs{2}.processFcns={};
net5.trainParam.epochs = localiterations;
net5.trainParam.showWindow = 0;

net6 = patternnet(numberofneuron);
    net6.inputs{1}.processFcns={};
 net6.outputs{2}.processFcns={};
net6.trainParam.epochs = localiterations;
net6.trainParam.showWindow = 0;

if usernumber>6
net7 = patternnet(numberofneuron);
    net7.inputs{1}.processFcns={};
 net7.outputs{2}.processFcns={};
net7.trainParam.epochs = localiterations;
net7.trainParam.showWindow = 0;


net8 = patternnet(numberofneuron);
    net8.inputs{1}.processFcns={};
 net8.outputs{2}.processFcns={};
net8.trainParam.epochs = localiterations;
net8.trainParam.showWindow = 0;

net9 = patternnet(numberofneuron);
    net9.inputs{1}.processFcns={};
 net9.outputs{2}.processFcns={};
net9.trainParam.epochs = localiterations;
net9.trainParam.showWindow = 0;

if usernumber>9
net10 = patternnet(numberofneuron);
    net10.inputs{1}.processFcns={};
 net10.outputs{2}.processFcns={};
net10.trainParam.epochs = localiterations;
net10.trainParam.showWindow = 0;

net11 = patternnet(numberofneuron);
    net11.inputs{1}.processFcns={};
 net11.outputs{2}.processFcns={};
net11.trainParam.epochs = localiterations;
net11.trainParam.showWindow = 0;

net12 = patternnet(numberofneuron);
    net12.inputs{1}.processFcns={};
 net12.outputs{2}.processFcns={};
net12.trainParam.epochs = localiterations;
net12.trainParam.showWindow = 0;

if usernumber>12
net13 = patternnet(numberofneuron);
    net13.inputs{1}.processFcns={};
 net13.outputs{2}.processFcns={};
net13.trainParam.epochs = localiterations;
net13.trainParam.showWindow = 0;

net14 = patternnet(numberofneuron);
    net14.inputs{1}.processFcns={};
 net14.outputs{2}.processFcns={};
net14.trainParam.epochs = localiterations;
net14.trainParam.showWindow = 0;

net15 = patternnet(numberofneuron);
    net15.inputs{1}.processFcns={};
 net15.outputs{2}.processFcns={};
net15.trainParam.epochs = localiterations;
net15.trainParam.showWindow = 0;
end
end
end
end


for i=1:1:iteration
    

for user=1:1:usernumber
    
   
 %%%%%%%%%%%%%%%%%%%%%%%%Set input of local FL model %%%%%%%%%%%%%%%%%%%%%%            
    x1=[trainingdata1(1+(user-1)*eachdatanumber:user*eachdatanumber,:);trainingdata2(1+(user-1)*eachdatanumber:user*eachdatanumber,:);...
        trainingdata3(1+(user-1)*eachdatanumber:user*eachdatanumber,:);trainingdata4(1+(user-1)*eachdatanumber:user*eachdatanumber,:);...
        trainingdata5(1+(user-1)*eachdatanumber:user*eachdatanumber,:);trainingdata6(1+(user-1)*eachdatanumber:user*eachdatanumber,:);...
        trainingdata7(1+(user-1)*eachdatanumber:user*eachdatanumber,:);trainingdata8(1+(user-1)*eachdatanumber:user*eachdatanumber,:);...
        trainingdata9(1+(user-1)*eachdatanumber:user*eachdatanumber,:);trainingdata10(1+(user-1)*eachdatanumber:user*eachdatanumber,:);];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 %%%%%%%%%%%%%%%%%%%%%%%% Set output of local FL model %%%%%%%%%%%%%%%%%%%%%%            
     y1=[traingnd1(1+(user-1)*eachdatanumber:user*eachdatanumber,:);traingnd2(1+(user-1)*eachdatanumber:user*eachdatanumber,:);...
        traingnd3(1+(user-1)*eachdatanumber:user*eachdatanumber,:);traingnd4(1+(user-1)*eachdatanumber:user*eachdatanumber,:);...
        traingnd5(1+(user-1)*eachdatanumber:user*eachdatanumber,:);traingnd6(1+(user-1)*eachdatanumber:user*eachdatanumber,:);...
        traingnd7(1+(user-1)*eachdatanumber:user*eachdatanumber,:);traingnd8(1+(user-1)*eachdatanumber:user*eachdatanumber,:);...
        traingnd9(1+(user-1)*eachdatanumber:user*eachdatanumber,:);traingnd10(1+(user-1)*eachdatanumber:user*eachdatanumber,:);];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

    clear netvaluable;
    Winstr1=strcat('net',int2str(user));
     eval(['netvaluable','=',Winstr1,';']);
    
if i > 1
   % Let global FL model to be the local FL model of each user, which is
   % equal to that the BS transmits the global FL model to the users
    netvaluable.IW{1,1}=wglobal;
    netvaluable.LW{2,1}=lwglobal;
     netvaluable.b{1,1}=bglobal;
     netvaluable.b{2,1}=obglobal;
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
      
end


oldnetvaluable=netvaluable;
[netvaluable,tr] =  train(netvaluable,x1',y1'); % Train local FL model.

if i==1
       wglobal=zeros(size(netvaluable.IW{1,1}));
    lwglobal=zeros(size(netvaluable.LW{2,1}));
    bglobal=zeros(size(netvaluable.b{1,1}));
    obglobal=zeros(size(netvaluable.b{2,1}));
end

% Record trained local FL model.

w(:,:,user)=netvaluable.IW{1,1};

lw(:,:,user)=netvaluable.LW{2,1};

     b(:,:,user)=netvaluable.b{1,1};
     ob(:,:,user)=netvaluable.b{2,1};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
  

if proposed==1
%%%%%%%%%%%%% Calculate the gradient of local FL model of each user%%%%%%%        
if i==1    
deviationw(:,:,user)=netvaluable.IW{1,1};
deviationlw(:,:,user)=netvaluable.LW{2,1};
deviationb(:,:,user)=netvaluable.b{1,1};
deviationob(:,:,user)=netvaluable.b{2,1};

else
deviationw(:,:,user)=netvaluable.IW{1,1}-oldnetvaluable.IW{1,1};
deviationlw(:,:,user)=netvaluable.LW{2,1}-oldnetvaluable.LW{2,1};
deviationb(:,:,user)=netvaluable.b{1,1}-oldnetvaluable.b{1,1};
deviationob(:,:,user)=netvaluable.b{2,1}-oldnetvaluable.b{2,1};
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    m_fH1 = [deviationw(:,:,user),deviationb(:,:,user)];  % coding 
%    
   [m_fHhat1, ~] = m_fQuantizeData(m_fH1, s_fRate, stSettings); % decoding 
   
     m_fH2 = [deviationlw(:,:,user),deviationob(:,:,user)]; % coding 
   
   [m_fHhat2, ~] = m_fQuantizeData(m_fH2, s_fRate, stSettings); % decoding 
  
     %%%%%%%%%%%%% Calculate the gradient after decoding %%%%%%%   
    deviationwnew(:,:,user)= m_fHhat1(:,1:length(netvaluable.IW{1,1}));                    
    deviationlwnew(:,:,user)= m_fHhat2(:,1:length(netvaluable.LW{2,1}));  
    deviationbnew(:,:,user)=m_fHhat1(:,length(netvaluable.IW{1,1})+1);
    deviationobnew(:,:,user)=m_fHhat2(:,length(netvaluable.LW{2,1})+1);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    
    
   %%%%%%%%%%%%% Calculate the global FL model after decoding %%%%%%%        
    if i==1
            w(:,:,user)=deviationwnew(:,:,user);

    
          lw(:,:,user)=deviationlwnew(:,:,user);
          b(:,:,user)=deviationbnew(:,:,user);
          ob(:,:,user)=deviationobnew(:,:,user);
    else
     w(:,:,user)=oldnetvaluable.IW{1,1}+deviationwnew(:,:,user);

    
    lw(:,:,user)=oldnetvaluable.LW{2,1}+deviationlwnew(:,:,user);
    b(:,:,user)=oldnetvaluable.b{1,1}+deviationbnew(:,:,user);
   ob(:,:,user)=oldnetvaluable.b{2,1}+deviationobnew(:,:,user);
    end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
end    
    
    
eval([Winstr1,'=','netvaluable',';']);

end


%%%%%%%% Global FL model update %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

wglobal=1/usernumber*sum(w,3);  % global training model
lwglobal=1/usernumber*sum(lw,3);  % global training model
bglobal=1/usernumber*sum(b,3);
obglobal=1/usernumber*sum(ob,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%% Calculate identification accuracy %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

[nn,mm]=max(net1(testdata(1:10000,:)'));
    
    oo=mm'-testgnd(1:10000,:);
    
    
    
    
    error(i)=length(find(oo~=0))/10000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end





    


end


end

plot(error);
