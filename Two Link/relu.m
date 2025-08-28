function y=relu(x)  %actually leaky ReLU
y=x;
for i=1:length(x)
if(x(i)>=0)
    y(i)=x(i);
else
    y(i)=0.1*x(i);
end
end