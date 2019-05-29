clear all
clc

datadir='D:\dataset\DIV_VAL_100'; %validation data directory
count=1;
scale=4; %scale factor
cnt=0;

f_lst=[];
f_lst=[f_lst; dir(fullfile(datadir, '*.png'))];
f_lst=[f_lst; dir(fullfile(datadir, '*.jpg'))];
f_lst=[f_lst; dir(fullfile(datadir, '*.bmp'))];
% f_lst = 3x1 struct

patch_size=32; %same as training patch size

num_lst=numel(f_lst); 
val_label_all=uint8(zeros(400000, patch_size, patch_size, 1)); % gt
val_patch_all=uint8(zeros(400000, patch_size, patch_size, 1)); % input 

for f_iter = 1:num_lst%numel(f_lst)
    f_iter
    f_info=f_lst(f_iter);
    if f_info.name=='.'
        continue;
    end
    f_path=fullfile(datadir,f_info.name);
    img_raw=imread(f_path);
    if size(img_raw, 3)==3
        img_raw=rgb2ycbcr(img_raw);
        img_raw=img_raw(:,:,1);
    %else
    %    img_raw=rgb2ycbcr(repmat(img_raw, [1 1 3]));
    end
    img_size=size(img_raw); width=img_size(2); height=img_size(1);
    img_raw=img_raw(1:height-mod(height,scale),1:width-mod(width,scale),:);
    img_size=size(img_raw); width=img_size(2); height=img_size(1);
    
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
            val_label_all(count,:,:,1)=patch;
            patch=img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR);
            patch=img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR);
            patch=img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR);
            val_patch_all(count,:,:,1)=patch;
            count=count+1;
            
            patch=imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:), 90);
            val_label_all(count,:,:,1)=patch;
            patch=imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 90);
            patch=imrotate(img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 90);
            patch=imrotate(img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 90);
            val_patch_all(count,:,:,1)=patch;
            count= count+1;
            
            
            patch=imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:), 180);
            val_label_all(count,:,:,1)=patch;
            patch=imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 180);
            patch=imrotate(img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 180);
            patch=imrotate(img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 180);
            val_patch_all(count,:,:,1)=patch;
            count= count+1;
%             
            patch=imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:), 270);
            val_label_all(count,:,:,1)=patch;
            patch=imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 270);
            patch=imrotate(img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 270);
            patch=imrotate(img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 270);
            val_patch_all(count,:,:,1)=patch;
            count= count+1;
            
            patch=fliplr(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:));
            val_label_all(count,:,:,1)=patch;
            patch=fliplr(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:));
            patch=fliplr(img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:));
            patch=fliplr(img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:));
            val_patch_all(count,:,:,1)=patch;
            count=count+1;
            
            patch=fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),90));
            val_label_all(count,:,:,1)=patch;
            patch=fliplr(imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),90));
            patch=fliplr(imrotate(img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),90));
            patch=fliplr(imrotate(img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),90));
            val_patch_all(count,:,:,1)=patch;
            count=count+1;
            
            patch=fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),180));
            val_label_all(count,:,:,1)=patch;
            patch=fliplr(imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),180));
            patch=fliplr(imrotate(img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),180));
            patch=fliplr(imrotate(img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),180));
            val_patch_all(count,:,:,1)=patch;
            count=count+1;
             
            patch=fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),270));
            val_label_all(count,:,:,1)=patch;
            patch=fliplr(imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),270));
            patch=fliplr(imrotate(img_3(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),270));
            patch=fliplr(imrotate(img_4(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),270));
            val_patch_all(count,:,:,1)=patch;
            count=count+1;
            
        end
    end
    
    cnt=cnt+1;
    if mod(cnt,100)==0
        display(100*cnt/numel(f_lst));display('percent complete');
    end
    
end
val_label_all=val_label_all(1:count-1,:,:,:);
val_patch_all=val_patch_all(1:count-1,:,:,:);

order=randperm(count-1);
val_label_all=val_label_all(order,:,:,:);
val_patch_all=val_patch_all(order,:,:,:);

patch_name='val.h5'; %save file name 'val.h5' 
h5create(patch_name,'/label_all',size(val_label_all),'Datatype','uint8');
h5write(patch_name,'/label_all',val_label_all);

h5create(patch_name,'/patch_all',size(val_patch_all),'Datatype','uint8');
h5write(patch_name,'/patch_all',val_patch_all);