dataDir = '91\';
mkdir('train_91_input');
mkdir('train_91_label');
folder_label = fullfile('train_91_label');
folder_input = fullfile('train_91_input');
count_input = 0;
count_label = 0;
count = 0;
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.png'))];

for f_iter = 1:numel(f_lst)

%     disp(f_iter);
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    img_raw = imread(f_path);
    img_raw = rgb2ycbcr(img_raw);
    img_raw = im2double(img_raw(:,:,1));
    
    img_size = size(img_raw);
    width = img_size(2);
    height = img_size(1);
    
    img_raw = img_raw(1:height-mod(height,12),1:width-mod(width,12),:);
    
    img_size = size(img_raw);
    patch_size = 32;
    stride = 14;
    x_size = (img_size(2)-patch_size)/stride+1;
    y_size = (img_size(1)-patch_size)/stride+1;
    
    img_2 = imresize(imresize(img_raw,1/2,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    img_3 = imresize(imresize(img_raw,1/3,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    img_4 = imresize(imresize(img_raw,1/4,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    
    for x = 0:x_size-1
        for y = 0:y_size-1
            x_coord = x*stride; y_coord = y*stride; 

           
            patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
            patch_name = sprintf('%s/%d',folder_label, count_label)
            save(sprintf('%s.mat', patch_name), 'patch', '-v6');
            count_label = count_label+1;

            patch = imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
            patch_name = sprintf('%s/%d',folder_input, count_input)
            save(sprintf('%s.mat', patch_name), 'patch', '-v6');
            count_input = count_input + 1;
            
            
        end
    end
    
    display(count);
    count = count + 1;
    
end