% Author: Samira Daneshgar-Asl, Ioannis Paraskevakos
% License: MIT
% Copyright: 2018-2019

function mosaic(FileName, FilePath, PredictedPath)
    File = fullfile(FilePath,FileName);
    [img, R] = geotiffread(File);

    patch_size=800;

    desired_row_size=(patch_size/2)*ceil(size(img,1)/(patch_size/2));
    desired_col_size=(patch_size/2)*ceil(size(img,2)/(patch_size/2));

    image = zeros(desired_row_size,desired_col_size,size(img,3),'int16');
    image(1:size(img,1),1:size(img,2),:) = img;

    a=1:patch_size/2:size(image, 1);
    b=1:patch_size/2:size(image, 2);
    row = 1:patch_size/2:a(1,end-1);
    col = 1:patch_size/2:b(1,end-1);
    A=zeros(a(1,end)+(patch_size/2)-1,b(1,end)+(patch_size/2)-1,'single');
    B=zeros(a(1,end)+(patch_size/2)-1,b(1,end)+(patch_size/2)-1,'single');

    files = dir(fullfile(PredictedPath,'data/WV_predicted', '*.tif'));
    files_ordered = natsortfiles({files.name});
    totalFiles = numel(files);

    k=1;
    for i=1:size(row,2)
        for j=1:size(col,2)
            I=imread(fullfile(PredictedPath, 'data/WV_predicted', files_ordered{1,k}));
            A(row(i):row(i)+patch_size-1,col(j):col(j)+patch_size-1)=I;
            B=max(A,B);
            if k~=totalFiles
                k=k+1;
            else
            end
        end
    end

    B(sum(image,3)==0)=0;
    xi = [col(1,1)-.5, col(1,end)+patch_size-1+.5];
    yi = [row(1,1)-.5, row(1,end)+patch_size-1+.5];
    [xlimits, ylimits] = intrinsicToWorld(R, xi, yi);
    subR = R;
    subR.RasterSize = size(B);
    subR.XLimWorld = sort(xlimits);
    subR.YLimWorld = sort(ylimits);
    info = geotiffinfo(File);
    writeFileName=[strtok(FileName, '.'),'-mask-predicted.tif'];
    geotiffwrite(writeFileName,B,subR,'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag,'TiffTags',struct('Compression',Tiff.Compression.None))
end