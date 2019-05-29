% make a test file format .mat

clear all
clc

datadir='D:\dataset\test\BSD200'; %test image directory
count=0;
scale=4; %scale factor 
testdir='D:\SRCNN\SRCNN_KERAS\test_file\bsd200_x4';
f_lst=[];

f_lst=[f_lst; dir(fullfile(datadir, '*.png'))];
f_lst=[f_lst; dir(fullfile(datadir, '*.jpg'))];
f_lst=[f_lst; dir(fullfile(datadir, '*.bmp'))];
% f_lst = 3x1 struct

num_lst=numel(f_lst); 

for f_iter = 1:num_lst%numel(f_lst)
    f_info=f_lst(f_iter);
    if f_info.name=='.'
        continue;
    end
    f_path=fullfile(datadir,f_info.name);
    img_raw=imread(f_path);
    
    if size(img_raw,3)==3
        img_raw=rgb2ycbcr(img_raw);
        img_raw=img_raw(:,:,1);
    else
        img_raw=rgb2ycbcr(repmat(img_raw,[1 1 3]));
        img_raw=img_raw(:,:,1);
    end
    
    img_size=size(img_raw); width=img_size(2); height=img_size(1);
    img_raw=img_raw(1:height-mod(height,scale),1:width-mod(width,scale),:);
    img_size=size(img_raw); width=img_size(2); height=img_size(1);
    
    img_2=imresize(imresize(img_raw,1/2, 'bicubic'),[img_size(1), img_size(2)], 'bicubic');
    img_3=imresize(imresize(img_raw,1/3, 'bicubic'),[img_size(1), img_size(2)], 'bicubic');
    img_4=imresize(imresize(img_raw,1/4, 'bicubic'),[img_size(1), img_size(2)], 'bicubic');
   

    patch_name=sprintf('%s/%d',testdir,count);
    save(patch_name,'img_raw');
    save(sprintf('%s_2',patch_name), 'img_2');
    save(sprintf('%s_3', patch_name), 'img_3');
    save(sprintf('%s_4', patch_name), 'img_4');
    
    count=count+1;
    display(count);
end
