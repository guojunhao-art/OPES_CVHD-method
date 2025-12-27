function [result] = sumhills(HILLS,cv,number)

for i=1:number
    if i==1
        result=HILLS(i,4)*exp(-(cv-(HILLS(i,2)-1)).^2/2/HILLS(i,3)^2);
    else
        result=result+HILLS(i,4)*exp(-(cv-(HILLS(i,2)-1)).^2/2/HILLS(i,3)^2);
    end
end
end