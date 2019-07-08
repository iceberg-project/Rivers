% Author: Samira Daneshgar-Asl
% License: MIT
% Copyright: 2018-2019

% when we have an 8-bit image with 3 bands and we want to save it as a multi-page

function multipagetiff(ReadImage, WriteDir)

    if ~exist(WriteDir, 'dir')
        mkdir(WriteDir);
    end

    image = geotiffread(ReadImage);
    writeFileName = strcat(WriteDir,'/multipage-',num2str(ReadImage));
    saveastiff(image,writeFileName);
end