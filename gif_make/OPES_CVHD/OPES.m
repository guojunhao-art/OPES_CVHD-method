function [results] = OPES(barrier,KERNELS,cv,n)
kbt=1;
bias_prefactor=barrier/kbt;
prob=0;
sum_hight=0;
sum_uprob=0;
eps=exp(-barrier/kbt/(1-1/bias_prefactor));
for i=1:n
    dist=(cv-(KERNELS(i,2)-1))./KERNELS(i,3);
    norm2=dist.*dist;
    val=KERNELS(i,4)*exp(-0.5*norm2);
    prob=prob+val;
    sum_hight=exp(KERNELS(i,5))+sum_hight;
    for j=1:n
        dist_i=((KERNELS(i,2)-1)-(KERNELS(j,2)-1))/KERNELS(j,3);
        norm2_i=dist_i*dist_i;
        sum_uprob=sum_uprob+KERNELS(j,4)*exp(-0.5*norm2_i);
    end
end
KDEnorm=sum_hight;
%KDEnorm_=sum_weights_;
prob=prob/KDEnorm;
Zed=sum_uprob/KDEnorm/i;
results=barrier+kbt*(1-1/bias_prefactor)*log(prob/Zed+eps);
end