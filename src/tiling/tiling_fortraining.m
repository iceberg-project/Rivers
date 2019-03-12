% This program prepares the data for training purpose.
% It tiles a 16-bit 8-band WV image and its corresponding 8-bit river mask to windows of 800 by 800 with steps of 100,
% and saves the multi-page format of the generated tils in the 'tiled multi-page image' and 'tiled multi-page river mask' folders.

clear all
clc

[FileName1, PathName1] = uigetfile('*.tif', 'Select a 16-bit 8-band WV image for training:');
image = imread(fullfile(PathName1, FileName1));

[FileName2, PathName2] = uigetfile('*.tif', 'Select the corresponding 8-bit river mask:');
river_mask = imread(fullfile(PathName2, FileName2));
river_mask(river_mask~=0)=255;

WriteDir1 = fullfile(pwd, 'tiled multi-page image');
if ~exist(WriteDir1, 'dir')
    mkdir(WriteDir1);
end

WriteDir2 = fullfile(pwd, 'tiled multi-page river mask');
if ~exist(WriteDir2, 'dir')
    mkdir(WriteDir2);
end

patch_size=800;

a=1:patch_size:size(image, 1);
b=1:patch_size:size(image, 2);

step=100;

k=1;
for row = 1:step:a(1,end-1)
    for col = 1:step:b(1,end-1)
        
        clear img_window
        clear river_mask_window
        img_window = image(row:row+patch_size-1, col:col+patch_size-1,:);
        river_mask_window=river_mask(row:row+patch_size-1, col:col+patch_size-1,:);
        if nnz(sum(img_window,3))== patch_size^2
            
            outputFileName1 = fullfile(WriteDir1, sprintf('%.02d.tif', k));
            saveastiff(img_window,outputFileName1);
            
            river_mask_Array(:,:,1)=river_mask_window;
            river_mask_Array(:,:,2)=imcomplement(river_mask_window);
            outputFileName2 = fullfile(WriteDir2, sprintf('%.02d.tif', k));
            saveastiff(river_mask_Array,outputFileName2);
            
            k=k+1;
        end
    end
end

