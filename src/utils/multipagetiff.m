% Author: Samira Daneshgar-Asl
% License: MIT
% Copyright: 2018-2019

% when we have an 8-bit image with 3 bands and we want to save it as a multi-page

function multipagetiff(ReadDir, WriteDir)

    if ~exist(WriteDir, 'dir')
        mkdir(WriteDir);
    end

    image_files = dir(fullfile(ReadDir, '*.tif'));
    totalFiles = numel(image_files);

    for i =1:totalFiles
        ReadImage = image_files(i).name;
        if isunix
            image = geotiffread(strcat(ReadDir,'/',ReadImage));
            writeFileName = strcat(WriteDir,'/',strtok(ReadImage,'.'), '-multipage.tif');
        elseif ispc
            image = geotiffread(strcat(ReadDir,'\',ReadImage));
            writeFileName = strcat(WriteDir,'\',strtok(ReadImage,'.'), '-multipage.tif');
        else
            disp 'Something went wrong';
        end

        saveastiff(image,char(writeFileName));
end
