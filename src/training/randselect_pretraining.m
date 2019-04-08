% Author: Samira Daneshgar-Asl
% License: MIT
% Copyright: 2018-2019

% This program randomly selects 80% of the multi-page tiles for training and it makes the 'data' directory which will be used by 'training.py'
% It renames and saves the training and test tiles of image in 'data\image_tiles_fortraining' and 'data\image_tiles_fortest', respectively.
% It renames and saves the training and test tiles of the river mask in 'data\mask_tiles_fortraining' and 'data\mask_tiles_fortest', respectively.

clear all
clc

WriteDir1 = fullfile(pwd, 'data\image_tiles_fortraining\');
if ~exist(WriteDir1, 'dir')
    mkdir(WriteDir1);
end

WriteDir2 = fullfile(pwd, 'data\image_tiles_fortest\');
if ~exist(WriteDir2, 'dir')
    mkdir(WriteDir2);
end

WriteDir3 = fullfile(pwd, 'data\mask_tiles_fortraining\');
if ~exist(WriteDir3, 'dir')
    mkdir(WriteDir3);
end

WriteDir4 = fullfile(pwd, 'data\mask_tiles_fortest\');
if ~exist(WriteDir4, 'dir')
    mkdir(WriteDir4);
end

FileList1 = dir(fullfile('tiling\tiled multi-page image', '*.tif'));
index    = randperm(numel(FileList1), floor(0.8*numel(FileList1)));
for k = 1:floor(0.8*numel(FileList1))
    movefile(fullfile('tiling\tiled multi-page image', FileList1(index(k)).name), fullfile(WriteDir1, sprintf('%02d.tif', k)));
    movefile(fullfile('tiling\tiled multi-page river mask', FileList1(index(k)).name), fullfile(WriteDir3, sprintf('%02d.tif', k)));
end

FileList2 = dir(fullfile('tiling\tiled multi-page image', '*.tif'));
for k = 1:numel(FileList2)
    movefile(['tiling\tiled multi-page image\' FileList2(k).name],[WriteDir2 'Test-' FileList2(k).name]);
    movefile(['tiling\tiled multi-page river mask\' FileList2(k).name], [WriteDir4 'Mask-Test-' FileList2(k).name]);
end

rmdir('tiling\tiled multi-page image','s')
rmdir('tiling\tiled multi-page river mask','s')

