%Link to my Video: 

%clear all;
%clear;
%clc;

% Reference: Lecture Video [NIC]

x1=-512:1:512;
x2=-512:1:512;

for i=1:length(x1)
    for j=1:length(x2)
        %Eggholder Function 
        %Desired Signal
        y(i,j)=-(x2(j)+47).*(sin(sqrt(abs((x1(i)/2)+(x2(j)+47)))))-(x1(i).*sin(sin(sqrt(abs((x1(i))-(x2(j)+47))))));
    end
end
mesh(x1,x2,y)

hidden_layers=3; %Using this link which says- The number of hidden neurons should be less than twice the size of the input layer % https://www.heatonresearch.com/2017/06/01/hidden-layers.html
w1=rand(hidden_layers,2);%Weights between input layer and the hidden layer
w2=rand(1,hidden_layers); %Weights between the output layer and the hidden layer

b1=rand(hidden_layers,1);%Biases of the hidden layer
b2=rand; %Bias of the output neuron

for iter=1:50
    
    for i=1:length(x1)
        for j=1:length(x2)
            a1=tanh(w1*[x1(i) x2(j)]'+b1); % Tan Sigmoid activation function has more flexibility compared to Log Sigmoid activation function
            y1(i,j)=(w2*a1)+b2; %approximated y1
            e(i,j)=y(i,j)-y1(i,j); %Actual-Approximated
            w2=w2+2*lr*e(i,j)*a1';
            b2=b2+2*lr*e(i,j);
            w1=w1+2*lr*e(i,j)*diag(1-a1.^2)*w2'*[x1(i) x2(j)];
            b1=b1+2*lr*e(i,j)*diag(1-a1.^2)*w2';   
            
        end
    end
    mesh(x1,x2,y1)
    xlabel('x')
    ylabel('y')
    zlabel('z')
    title('Eggholder Function')
    disp(iter)
end

figure
mesh(x1,x2,y1)
axis([-512 512 -512 512 -1000 1000]);

 %%
 % Random Selection of Test Points

t_x1=-512:0.01:512;
t_x2=-512:0.01:512;
test_x1=[];
test_x2=[];

while length(test_x1)<40
    pos=randi(length(t_x1));
    card=t_x1(pos);
    if isinteger(card)
        continue
    else
        test_x1=[test_x1,card];
    end
end

while length(test_x2)<40
    pos=randi(length(t_x2));
    card=t_x2(pos);
    if isinteger(card)
        continue
    else
        test_x2=[test_x2,card];
    end
end

%% Calculating the Mean Square Error
MSE=0;

for i=1:40
    y=-(test_x2(i)+47).*(sin(sqrt(abs((test_x1(i)/2)+(test_x2(i)+47)))))-(test_x1(i).*sin(sin(sqrt(abs((test_x1(i))-(test_x2(i)+47))))));
    a1=tanh(w1*[test_x1(i) test_x2(i)]'+b1);
    y1=(w2*a1)+b2;
    MSE=MSE+(y-y1)^2
end

MSE=MSE/40

 