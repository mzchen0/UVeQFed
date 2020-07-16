# UVeQFed: Universal Vector Quantization for Federated Learning
Pleasue use Matlab 2018b or above to run the simulations.

The simulations consist of three main files: MINSTIID.m, MINSTnonIID.m, and CIFAR.m.

MINSTIID.m and MINSTnonIID.m are the code that focuses on the use of federated learning for handwritten digit identifications. In MINSTIID, the data of each user is IID while in MINSTnonIID, the dataset is non-IID.

To run MINSTIID.m or MINSTnonIID.m, one must put all the code files into one folder. Then, one can directly run MINSTIID.m or MINSTnonIID.m. 


CIFAR is used for image identification.  One can change the data distribution using our predefined function GetUnbalancedCIFAR.m. Before running the code CIFAR.m, one must first run the code DownloadCIFAR10.m to downlowd the dataset of CIFAR. After that, one must run the code 
% if ~exist('cifar10Train','dir')
%     disp('Saving the Images in folders. This might take some time...');    
%     saveCIFAR10AsFolderOfImages('cifar-10-batches-mat', pwd, true);
% end
to classify the CIFAR dataset. Then, one can directly run the code CIFAR.m. 


In coding setting, stSettings.type determine the coding method. Fore example, stSettings.type=2 implies that Dithered 2-D lattice quantization method is used for coding while stSettings.type=3 implies that Dithered scalar quantization is used for coding. s_fRate determines the number of bits used to represent one element in the local FL model vector.
