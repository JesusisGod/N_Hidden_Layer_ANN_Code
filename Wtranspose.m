
function WT=Wtranspose(W,row,col)
WT=zeros(row*col,1);
for i=1:row
    stp=1+(i-1)*col;
    for j=1:col
        WT(stp+(j-1))=W(i+(j-1)*row);
    end
end