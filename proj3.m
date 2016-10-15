%-----------------------------------------------------------------------------------------------------------%
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images - Training Set
fp1 = fopen('C:\Work\COURSES\SEM-1\MACHINE LEARNIN\ML Project-3\train-images.idx3-ubyte', 'rb');
assert(fp1 ~= -1, ['Could not open ', 'C:\Work\COURSES\SEM-1\MACHINE LEARNIN\ML Project-3\train-images.idx3-ubyte', '']);
magic = fread(fp1, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', 'C:\Work\COURSES\SEM-1\MACHINE LEARNIN\ML Project-3\train-images.idx3-ubyte', '']);
numImages = fread(fp1, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp1, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp1, 1, 'int32', 0, 'ieee-be');
images = fread(fp1, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
%images = permute(images,[2 1 3]);
fclose(fp1);
% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

%-----------------------------------------------------------------------------------------------------------%
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images - Test Set
fp1_T = fopen('C:\Work\COURSES\SEM-1\MACHINE LEARNIN\ML Project-3\t10k-images.idx3-ubyte', 'rb');
assert(fp1_T ~= -1, ['Could not open ', 'C:\Work\COURSES\SEM-1\MACHINE LEARNIN\ML Project-3\t10k-images.idx3-ubyte', '']);
magic_T = fread(fp1_T, 1, 'int32', 0, 'ieee-be');
assert(magic_T == 2051, ['Bad magic number in ', 'C:\Work\COURSES\SEM-1\MACHINE LEARNIN\ML Project-3\t10k-images.idx3-ubyte', '']);
numImages_T = fread(fp1_T, 1, 'int32', 0, 'ieee-be');
numRows_T = fread(fp1_T, 1, 'int32', 0, 'ieee-be');
numCols_T = fread(fp1_T, 1, 'int32', 0, 'ieee-be');
T_images = fread(fp1_T, inf, 'unsigned char');
T_images = reshape(T_images, numCols_T, numRows_T, numImages_T);
%T_images = permute(T_images,[2 1 3]);
fclose(fp1_T);
% Reshape to #pixels x #examples
T_images = reshape(T_images, size(T_images, 1) * size(T_images, 2), size(T_images, 3));
% Convert to double and rescale to [0,1]
T_images = double(T_images) / 255;

%-----------------------------------------------------------------------------------------------------------%

%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images
fp2 = fopen('C:\Work\COURSES\SEM-1\MACHINE LEARNIN\ML Project-3\train-labels.idx1-ubyte', 'rb');
assert(fp2 ~= -1, ['Could not open ', 'C:\Work\COURSES\SEM-1\MACHINE LEARNIN\ML Project-3\train-labels.idx1-ubyte', '']);
magic = fread(fp2, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', 'C:\Work\COURSES\SEM-1\MACHINE LEARNIN\ML Project-3\train-labels.idx1-ubyte', '']);
numLabels = fread(fp2, 1, 'int32', 0, 'ieee-be');
labels = fread(fp2, inf, 'unsigned char');
assert(size(labels,1) == numLabels, 'Mismatch in label count');
fclose(fp2);

%-----------------------------------------------------------------------------------------------------------%

%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images-Test Set
fp2_T = fopen('C:\Work\COURSES\SEM-1\MACHINE LEARNIN\ML Project-3\t10k-labels.idx1-ubyte', 'rb');
assert(fp2_T ~= -1, ['Could not open ', 'C:\Work\COURSES\SEM-1\MACHINE LEARNIN\ML Project-3\t10k-labels.idx1-ubyte', '']);
magic2_T = fread(fp2_T, 1, 'int32', 0, 'ieee-be');
assert(magic2_T == 2049, ['Bad magic number in ', 'C:\Work\COURSES\SEM-1\MACHINE LEARNIN\ML Project-3\t10k-labels.idx1-ubyte', '']);
numLabels2_T = fread(fp2_T, 1, 'int32', 0, 'ieee-be');
T_labels = fread(fp2_T, inf, 'unsigned char');
assert(size(T_labels,1) == numLabels2_T, 'Mismatch in label count');
fclose(fp2_T);

%-----------------------------------------------------------------------------------------------------------%

% train the logistic regresion model for digit classification
eeta = 0.001;
N = 500;
n_out = 10;
n_in=784;

W_lr = rand([n_out, n_in+1]);
training_data = [images;ones(1,60000)];

tk = eye(10,10);

for i = 1:N
    for j = 1:size(training_data,2)
        train_img = training_data(:,j);
        train_lab = labels(j);
        yk = exp(W_lr * train_img)./sum(exp(W_lr * train_img));
        Err_x = (yk - tk(:, train_lab+1))*train_img';
        W_lr = W_lr-(eeta*Err_x);
    end
end

test_data = [T_images;ones(1,10000)];

cnt =0;
for l= 1:size(test_data,2)
    test_img = test_data(:,l);
    test_lab = T_labels(l);
    ak_T = W_lr * test_img;
    yk_T = exp(ak_T)./sum(exp(ak_T));
    [m_yk,I_yk] = max(yk_T); % Stores the value and the row number of the max value of yk.
    if ((I_yk-1)==test_lab) % Corresponds the row number-1 (Starts from 0,1,2...) to the label to check if true
        cnt=cnt+1;
    end
end
acc = cnt/size(T_labels,1);


UbitName = 'rnayak';
personNumber = 50169647;
Wlr = (W_lr(:,1:784))';
blr = (W_lr(:,785:785))';

%Single Neural Network

training_images= [images;ones(1,60000)];

W1_nn= rand(10,785)*0.065;
W2_nn= rand(10,11)*0.065;
h='sigmoid';
N_nn =500;
eeta_nn=0.0072;
tk_nn= eye(10,10);

for p= 1:N_nn
    for q= 1:size(training_images,2)
         nn_train_img = training_images(:,q);
         nn_train_label = labels(q);
         
         z1= 1./(1+(exp(-1*(W1_nn *nn_train_img))));
         zk= vertcat(z1,1);
         z_sig= zk .*(1-zk);
        
         yk_nn = exp(W2_nn*zk)/sum(exp(W2_nn*zk));
         
         dk= yk_nn-tk_nn(:, nn_train_label+1);
         dJ= z_sig .* (W2_nn'*dk);
         dJ(11,:)= [];
         
         Err1_nn = dJ * nn_train_img';
         Err2_nn = dk * zk';
         
         W1_nn = W1_nn - (eeta_nn * Err1_nn);
         W2_nn = W2_nn - (eeta_nn * Err2_nn);
         
    end
end

cnt_nn=0;
test_images= [T_images;ones(1,10000)];

 for q= 1:size(test_images,2)
         nn_test_img = test_images(:,q);
         nn_test_label = T_labels(q);
         
         T_z1= 1./(1+(exp(-1*(W1_nn *nn_test_img))));
         T_zk= vertcat(T_z1,1);
                 
         T_yk_nn = exp(W2_nn*T_zk)/sum(exp(W2_nn*T_zk));
         
         [nn_max_yk,nn_I_yk]= max(T_yk_nn);
         if((nn_I_yk-1)== nn_test_label)
             cnt_nn= cnt_nn+1;
         end
end
       
acc_nn= cnt_nn/size(T_labels,1);

Wnn1 = (W1_nn(:,1:784))';
Wnn2 = (W2_nn(:,1:10))';
bnn1 = (W1_nn(:,785:785))';
bnn2 = (W2_nn(:,11:11))';



save('proj3.mat', 'UbitName', 'personNumber', 'Wlr','blr', 'Wnn1','Wnn2','bnn1','bnn2','h');
    
        