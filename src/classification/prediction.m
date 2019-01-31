clc
clear all
 
[FileName, PathName] = uigetfile('*.tif', 'Select an image file:');
File = fullfile(PathName, FileName);
[I, R] = geotiffread(File);
info = geotiffinfo(File);
Image=I;
clear I;

%% Loading pretrained DenseNet201 from Training program
load('net_0.90_4_6Epochs.mat')

%% Calculating probabilities with steps of 56
N = 224;
step=56;
a=1:N:size(Image, 1);
b=1:N:size(Image, 2);
for row = 1:step:a(1,end-1)
    for col = 1:step:b(1,end-1)
        r = (row - 1) / step + 1;
        c = (col - 1) / step + 1;
        clear imgWindow
        clear imgWindow_1       
        imgWindow = Image(row:row+N-1, col:col+N-1,:);
        imgWindow_1=sum(imgWindow,3);
        if nnz(~imgWindow_1)>0
            probss(r,c,1:4)=NaN;            
        else
            [YPredd(r,c),probss(r,c,:)] = classify(net,imgWindow);                             
        end
    end
end

%% Calculating average probabilities
probss_1=probss;

for k=1:4
    for i=1:size(probss_1,1)-3
        for j=1:size(probss_1,2)-3
            probss_2(i,j,k)=nanmean(nanmean(probss_1(i:i+3,j:j+3,k)));
        end
    end
end    

%% Generating probability ASCII files importable to ArcGIS
[x,y] = pix2map(info.RefMatrix, a(1,end)-1, 1);

probss_3=single(NaN(size(probss_2, 1)+6,size(probss_2, 2)+6,size(probss_2, 3)));
probss_3(4:size(probss_2, 1)+3,4:size(probss_2, 2)+3,:)=probss_2*100;
probss_3(isnan(probss_3(:,:,:)))=-9999;

fileID = fopen('Crevasse.txt','wt');
fprintf(fileID,'NCOLS %i\r\n',size(probss_3,2))
fprintf(fileID,'NROWS %i\r\n',size(probss_3,1))
fprintf(fileID,'XLLCORNER %2f\r\n',x)
fprintf(fileID,'YLLCORNER %2f\r\n',y)
fprintf(fileID,'CELLSIZE %i\r\n',112)
fprintf(fileID,'NODATA_VALUE %i\r\n',-9999)
for k=1:size(probss_3,1)
    fprintf(fileID,'%7.1f\t',probss_3(k,:,1));
    fprintf(fileID,'\n');
end
fclose(fileID);


fileID = fopen('IceSlush.txt','wt');
fprintf(fileID,'NCOLS %i\r\n',size(probss_3,2))
fprintf(fileID,'NROWS %i\r\n',size(probss_3,1))
fprintf(fileID,'XLLCORNER %2f\r\n',x)
fprintf(fileID,'YLLCORNER %2f\r\n',y)
fprintf(fileID,'CELLSIZE %i\r\n',112)
fprintf(fileID,'NODATA_VALUE %i\r\n',-9999)
for k=1:size(probss_3,1)
    fprintf(fileID,'%7.1f\t',probss_3(k,:,2));
    fprintf(fileID,'\n');
end
fclose(fileID);


fileID = fopen('LargeRiver.txt','wt');
fprintf(fileID,'NCOLS %i\r\n',size(probss_3,2))
fprintf(fileID,'NROWS %i\r\n',size(probss_3,1))
fprintf(fileID,'XLLCORNER %2f\r\n',x)
fprintf(fileID,'YLLCORNER %2f\r\n',y)
fprintf(fileID,'CELLSIZE %i\r\n',112)
fprintf(fileID,'NODATA_VALUE %i\r\n',-9999)
for k=1:size(probss_3,1)
    fprintf(fileID,'%7.1f\t',probss_3(k,:,3));
    fprintf(fileID,'\n');
end
fclose(fileID);


fileID = fopen('SmallRiver.txt','wt');
fprintf(fileID,'NCOLS %i\r\n',size(probss_3,2))
fprintf(fileID,'NROWS %i\r\n',size(probss_3,1))
fprintf(fileID,'XLLCORNER %2f\r\n',x)
fprintf(fileID,'YLLCORNER %2f\r\n',y)
fprintf(fileID,'CELLSIZE %i\r\n',112)
fprintf(fileID,'NODATA_VALUE %i\r\n',-9999)
for k=1:size(probss_3,1)
    fprintf(fileID,'%7.1f\t',probss_3(k,:,4));
    fprintf(fileID,'\n');
end
fclose(fileID);

load chirp
sound(y,Fs)        
        
