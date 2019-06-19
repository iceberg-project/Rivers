% Author: Samira Daneshgar-Asl
% License: MIT
% Copyright: 2018-2019

% when we have an 8-bit image with 3 bands and we want to save it as a multi-page
clear all
clc

WriteDir = fullfile(pwd, '8bit-3bands Multi-Page Images');
if ~exist(WriteDir, 'dir')
    mkdir(WriteDir);
end

ReadDir = fullfile(pwd, '8bit-3bands Images');
files1 =dir(ReadDir);
files1(1:2) = [];
totalFiles = numel(files1);

for i =1:totalFiles
    Fileaddress{i,1}=strcat(ReadDir,'\',files1(i).name);
    file{i} = geotiffread(Fileaddress{i,1});  
    writeFileName = strcat(WriteDir,'\multipage-',num2str(files1(i).name));
    saveastiff(file{i},writeFileName);
    cd(ReadDir) % return to actualFile folder
end

