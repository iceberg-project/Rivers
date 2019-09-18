% Author: Samira Daneshgar-Asl
% License: MIT
% Copyright: 2018-2019

% This program prepares the data for training purpose.
% It tiles an 8-bit 3-band WV image and its corresponding 8-bit river mask to windows of 800 by 800 with steps of 400,
% and saves the multi-page format of the generated tils in the 'tiled multi-page image' and 'tiled multi-page river mask' folders.

function tiling_pretraining(Image, RiverMask, patch_size, step, OutputPath)
    
    if ~isstring(Image)
        error('Error. FileName1 must be a string.')
    end
    if ~isstring(RiverMask)
        error('Error. FileName2 must be a string.')
    end
    tmp_filename = split(Image,"/");
    PathName1 = join(tmp_filename(1:end-1),"/");
    FileName1 = tmp_filename(end);
    image = imread(fullfile(PathName1, FileName1));
    
    tmp_filename = split(RiverMask,"/");
    PathName2 = join(tmp_filename(1:end-1),"/");
    FileName2 = tmp_filename(end);
    river_mask = imread(fullfile(PathName2, FileName2));
    
    river_mask(river_mask~=0)=255;
    tmp_folder = split(FileName1,".");
    WriteDir1 = fullfile(OutputPath, 'tiled-multi-page-image', tmp_folder(1));
    if ~exist(WriteDir1, 'dir')
        mkdir(WriteDir1);
    end

    WriteDir2 = fullfile(OutputPath, 'tiled-multi-page-river-mask', tmp_folder(1));
    if ~exist(WriteDir2, 'dir')
        mkdir(WriteDir2);
    end

    a=1:patch_size:size(image, 1);
    b=1:patch_size:size(image, 2);

    k=1;
    for row = 1:step:a(1,end-1)
        for col = 1:step:b(1,end-1)
        
            clear img_window
            clear river_mask_window
            img_window = image(row:row+patch_size-1, col:col+patch_size-1,:);
            river_mask_window=river_mask(row:row+patch_size-1, col:col+patch_size-1,:);
            if nnz(sum(img_window,3))== patch_size^2
            
                outputFileName1 = fullfile(WriteDir1, sprintf('%.02d.tif', k));
                saveastiff(img_window,char(outputFileName1));
            
                river_mask_Array(:,:,1)=river_mask_window;
                river_mask_Array(:,:,2)=imcomplement(river_mask_window);
                outputFileName2 = fullfile(WriteDir2, sprintf('%.02d.tif', k));
                saveastiff(river_mask_Array,char(outputFileName2));
            
                k=k+1;
            end
        end
    end
end

