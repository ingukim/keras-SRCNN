clear all
clc

datadir='D:\dataset\train\291'; %traing file directory
count=1;
scale=2; %for scale factor x2, x3, x4
cnt=0;

f_lst=[];
f_lst=[f_lst; dir(fullfile(datadir, '*.png'))];
f_lst=[f_lst; dir(fullfile(datadir, '*.jpg'))];
f_lst=[f_lst; dir(fullfile(datadir, '*.bmp'))];
% f_lst = 3x1 struct

patch_size=32;  %patch size 

num_lst=numel(f_lst); %numel->배열 요소의 개수를 반환 return the number of array component
label_all=uint8(zeros(2000000, patch_size, patch_size, 1)); % (HR)for ground truth array size 
patch_all=uint8(zeros(2000000, patch_size, patch_size, 1)); % (LR)for low-resolution array size

for f_iter = 1:num_lst%numel(f_lst)
    f_iter
    f_info=f_lst(f_iter);
    if f_info.name=='.'
        continue;
    end
    f_path=fullfile(datadir,f_info.name);
    img_raw=imread(f_path);
    img_raw=rgb2ycbcr(img_raw);
    img_raw=img_raw(:,:,1);
    img_size=size(img_raw); width=img_size(2); height=img_size(1);
    img_raw=img_raw(1:height-mod(height,scale),1:width-mod(width,scale),:);
    img_size=size(img_raw); width=img_size(2); height=img_size(1);
    
    %making input data , HR-> bicubic -> upsampling bicubic for input data
    
    img_2=imresize(imresize(img_raw,1/2, 'bicubic'),[img_size(1), img_size(2)], 'bicubic');
    img_3=imresize(imresize(img_raw,1/3, 'bicubic'),[img_size(1), img_size(2)], 'bicubic');
    img_4=imresize(imresize(img_raw,1/4, 'bicubic'),[img_size(1), img_size(2)], 'bicubic');
   
    
    img_HR=size(img_raw);
    patch_HR=patch_size; 
    stride_HR=patch_size; 
    x_size=(img_HR(2)-patch_HR)/stride_HR+1;
    y_size=(img_HR(1)-patch_HR)/stride_HR+1;
    
    for y=0:y_size-1
        for x=0:x_size-1
            x_coord=x*stride_HR; y_coord=y*stride_HR;
            
            patch=img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:);
            label_all(count,:,:,1)=patch;
            patch=img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR);
            patch=img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR);
            patch=img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR);
            patch_all(count,:,:,1)=patch;
            count= count+1;
            
            patch=imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:), 90);
            label_all(count,:,:,1)=patch;
            patch=imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 90);
            patch=imrotate(img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 90);
            patch=imrotate(img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 90);
            patch_all(count,:,:,1)=patch;
            count= count+1;
            
            patch=imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:), 180);
            label_all(count,:,:,1)=patch;
            patch=imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 180);
            patch=imrotate(img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 180);
            patch=imrotate(img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 180);
            patch_all(count,:,:,1)=patch;
            count= count+1;
          
            patch=imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:), 270);
            label_all(count,:,:,1)=patch;
            patch=imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 270);
            patch=imrotate(img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 270);
            patch=imrotate(img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 270);
            patch_all(count,:,:,1)=patch;
            count= count+1;
            
            patch=fliplr(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:));
            label_all(count,:,:,1)=patch;
            patch=fliplr(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:));
            patch=fliplr(img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:));
            patch=fliplr(img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:));
            patch_all(count,:,:,1)=patch;
            count=count+1;
            
            patch=fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),90));
            label_all(count,:,:,1)=patch;
            patch=fliplr(imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),90));
            patch=fliplr(imrotate(img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),90));
            patch=fliplr(imrotate(img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),90));
            patch_all(count,:,:,1)=patch;
            count=count+1;
            
            patch=fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),180));
            label_all(count,:,:,1)=patch;
            patch=fliplr(imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),180));
            patch=fliplr(imrotate(img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),180));
            patch=fliplr(imrotate(img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),180));
            patch_all(count,:,:,1)=patch;
            count=count+1;
            
            patch=fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),270));
            label_all(count,:,:,1)=patch;
            patch=fliplr(imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),270));
            patch=fliplr(imrotate(img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),270));
            patch=fliplr(imrotate(img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),270));
            patch_all(count,:,:,1)=patch;
            count=count+1;
        end
    end
    
    cnt=cnt+1;
    if mod(cnt,100)==0
        display(100*cnt/numel(f_lst));display('percent complete');
    end
    
end
label_all=label_all(1:count-1,:,:,:);
patch_all=patch_all(1:count-1,:,:,:);

order=randperm(count-1);
label_all=label_all(order,:,:,:);
patch_all=patch_all(order,:,:,:);

patch_name='train.h5'; %save file name train.h5 with patch name ['label_all' for HR , 'patch_all' for LR]
h5create(patch_name,'/label_all',size(label_all),'Datatype','uint8');
h5write(patch_name,'/label_all',label_all);

h5create(patch_name,'/patch_all',size(patch_all),'Datatype','uint8');
h5write(patch_name,'/patch_all',patch_all);

    