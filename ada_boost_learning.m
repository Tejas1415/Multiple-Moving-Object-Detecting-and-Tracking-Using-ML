%% training a adaboost classifier
close all; clear; clc;
% Reading annotations file
cd New_DB;
fileID = fopen('MVI_20011.xml','r');
A = fscanf(fileID,'%c');
idcs = strfind(A,['num=']);
blue = [255,20,147];
n_indx = 1;

% for loop over each video frame
for l_indx = 1:length(idcs)
    % searching for annotations in each frame
    if(l_indx < 10)
        str1 =  '0000';
    elseif(l_indx < 100)
        str1 =  '000';
    else
        str1 = '00';
    end
    if(l_indx ~= 664)
    b_str = A(idcs(l_indx):idcs(l_indx+1));
    else
        b_str = A(idcs(l_indx):end);
    end
    idx_targets = strfind(b_str,'<target id=');
    num_obj = length(idx_targets);

    BB_box_num = zeros(num_obj,4);
    for box_i = 1:num_obj
        if(box_i ~= num_obj)
        tmp_str = b_str(idx_targets(box_i):idx_targets(box_i+1));
        else
            tmp_str = b_str(idx_targets(box_i):end);
        end
        b_indx1 = strfind(tmp_str,'<box left="');
        b_indx2 = strfind(tmp_str,'top="');
        b_indx3 = strfind(tmp_str,' width=');
        b_indx4 = strfind(tmp_str,' height=');
        b_indx5 = strfind(tmp_str,'<attribute');
        BB_box_num(box_i,1) = str2num(tmp_str(b_indx1+length('<box left="'):b_indx2-3));
        BB_box_num(box_i,2) = str2num(tmp_str(b_indx2+length('top="'):b_indx3-2));
        BB_box_num(box_i,3) = str2num(tmp_str(b_indx3+length(' width="'):b_indx4-2));
        BB_box_num(box_i,4) = str2num(tmp_str(b_indx4+length(' height="'):b_indx5-18));
    end
    % reading image
    img_file_name = ['img',str1,num2str(l_indx),'.jpg'];
    folder = 'MVI_20011';
    fullname = fullfile(folder,img_file_name);
    Currentframe = imread(fullname);
    for box_i = 1:1:num_obj
        crop_img = imcrop(Currentframe,BB_box_num(box_i,:));
%         imshow(crop_img);
        
        cell_num1 = (length(crop_img(:,1,1)));
        cell_num2 = (length(crop_img(1,:,1)));
        cell_num1 = floor(cell_num1/2);
        cell_num2 = floor(cell_num2/2);
        tmp = [extractHOGFeatures(crop_img,'numbins',8,'cellsize',[cell_num1,cell_num2])];
        HOG_feature(box_i, :) = tmp;
        HOG_feature(box_i, :) = (HOG_feature(box_i, :) );
        colrhist_feature(box_i,:) = [imhist(crop_img(:,:,1),8);...
            imhist(crop_img(:,:,2), 8);imhist(crop_img(:,:,2), 8)]';
        centroid_feat(box_i,:) = [BB_box_num(box_i,1)+(BB_box_num(box_i,1)/2) ,BB_box_num(box_i,3)+(BB_box_num(box_i,4)/2)];
    end
    
     %Displaying the output    
%         shapeInserter = vision.ShapeInserter('BorderColor','Custom',...
%             'CustomBorderColor',blue);
%         out1    = step(shapeInserter, Currentframe, int32(BB_box_num));
%         imshow(out1)
    if(l_indx > 1)
        NUM_OF_PAST_BLOBS = length(past_colrhist_feature(:,1)) ;
         if(NUM_OF_PAST_BLOBS > 2)
             NUM_OF_PAST_BLOBS = NUM_OF_PAST_BLOBS;
         end
         for indx1 = 1:num_obj
             for indx2 = 1:NUM_OF_PAST_BLOBS
            sim_values(indx1,indx2,1) = pdist2(past_HOG_features(indx2,:),HOG_feature(indx1,:));
            sim_values(indx1,indx2,2) = pdist2(past_colrhist_feature(indx2,:),colrhist_feature(indx1,:));
%             sim_values(indx1,indx2,3) =  sqrt(sum((past_centroid_feat(indx2,:) - centroid_feat(indx2,:)) .^ 2));
%             sim_values(indx1,indx2,3) = corrcoef(past_hus_feature(indx2,:),hus_feature(indx1,:));
             end
         end
        sim_values(:,:,1) = (sim_values(:,:,1))./max((sim_values(:,:,1)));
         sim_values(:,:,2) = (sim_values(:,:,2))./max((sim_values(:,:,2)));
         
         if(mod(l_indx,6) == 0)
         X(n_indx,:) = sim_values(6,6,:);
         Y(n_indx) = 1;
         elseif(mod(l_indx,6) == 1)
         X(n_indx,:) = sim_values(3,3,:);
         Y(n_indx) = 1;
         elseif(mod(l_indx,6) == 2)
         X(n_indx,:) = sim_values(1,1,:);
         Y(n_indx) = 1;
         elseif(mod(l_indx,6) == 4)
         X(n_indx,:) = sim_values(1,6,:);
         Y(n_indx) = 0;
         elseif(mod(l_indx,6) == 5)
         X(n_indx,:) = sim_values(3,4,:);
         Y(n_indx) = 0;
         elseif(mod(l_indx,6) == 6)
         X(n_indx,:) = sim_values(6,1,:);
         Y(n_indx) = 0;
         end

         clear past_HOG_features;
         clear past_colrhist_feature;
         clear sim_values;
    end
    
    clear BB_box_num
    
    if(l_indx > 1)
    past_HOG_features = HOG_feature;
    past_colrhist_feature = colrhist_feature;
    past_centroid_feat = centroid_feat;
    clear HOG_feature;
    clear colrhist_feature;
%     past_hus_feature = hus_feature;
    else
        past_HOG_features = zeros(1,32);
        past_colrhist_feature = zeros(1,24);
        past_centroid_feat = zeros(1,2);
    end
    n_indx = n_indx + 1;
end


%% Training part
         Mdl_ada = fitcensemble(X,Y,'Method','AdaBoostM1');
         
%% Testing part
fileID = fopen('MVI_20012.xml','r');
A = fscanf(fileID,'%c');
idcs = strfind(A,['num=']);
n_indx = 1;
for  l_indx = 1:100
    % searching for annotations in each frame
    if(l_indx < 10)
        str1 =  '0000';
    elseif(l_indx < 100)
        str1 =  '000';
    else
        str1 = '00';
    end
    if(l_indx ~= length(idcs))
    b_str = A(idcs(l_indx):idcs(l_indx+1));
    else
        b_str = A(idcs(l_indx):end);
    end
    idx_targets = strfind(b_str,'<target id=');
    num_obj = length(idx_targets);

    BB_box_num = zeros(num_obj,4);
    for box_i = 1:num_obj
        if(box_i ~= num_obj)
        tmp_str = b_str(idx_targets(box_i):idx_targets(box_i+1));
        else
            tmp_str = b_str(idx_targets(box_i):end);
        end
        b_indx1 = strfind(tmp_str,'<box left="');
        b_indx2 = strfind(tmp_str,'top="');
        b_indx3 = strfind(tmp_str,' width=');
        b_indx4 = strfind(tmp_str,' height=');
        b_indx5 = strfind(tmp_str,'<attribute');
        BB_box_num(box_i,1) = str2num(tmp_str(b_indx1+length('<box left="'):b_indx2-3));
        BB_box_num(box_i,2) = str2num(tmp_str(b_indx2+length('top="'):b_indx3-2));
        BB_box_num(box_i,3) = str2num(tmp_str(b_indx3+length(' width="'):b_indx4-2));
        BB_box_num(box_i,4) = str2num(tmp_str(b_indx4+length(' height="'):b_indx5-18));
    end
    % reading image
    img_file_name = ['img',str1,num2str(l_indx),'.jpg'];
    folder = 'MVI_20012';
    fullname = fullfile(folder,img_file_name);
    Currentframe = imread(fullname);
    for box_i = 1:1:num_obj
        crop_img = imcrop(Currentframe,BB_box_num(box_i,:));
%         imshow(crop_img);
        
        cell_num1 = (length(crop_img(:,1,1)));
        cell_num2 = (length(crop_img(1,:,1)));
        cell_num1 = floor(cell_num1/2);
        cell_num2 = floor(cell_num2/2);
        tmp = [extractHOGFeatures(crop_img,'numbins',8,'cellsize',[cell_num1,cell_num2])];
        HOG_feature(box_i, :) = tmp;
        HOG_feature(box_i, :) = (HOG_feature(box_i, :) );
        colrhist_feature(box_i,:) = [imhist(crop_img(:,:,1),8);...
            imhist(crop_img(:,:,2), 8);imhist(crop_img(:,:,2), 8)]';
        colrhist_feature(box_i,:) = (colrhist_feature(box_i, :) );
        centroid_feat(box_i,:) = [BB_box_num(box_i,1)+(BB_box_num(box_i,1)/2) ,BB_box_num(box_i,3)+(BB_box_num(box_i,4)/2)];
    end
    
    
     %Displaying the output    
        shapeInserter = vision.ShapeInserter('BorderColor','Custom',...
            'CustomBorderColor',blue);
        out1    = step(shapeInserter, Currentframe, int32(BB_box_num));
        
       
    if(l_indx > 1)
        NUM_OF_PAST_BLOBS = length(past_colrhist_feature(:,1)) ;

         for indx1 = 1:num_obj
             for indx2 = 1:NUM_OF_PAST_BLOBS
            sim_values(indx1,indx2,1) = pdist2(past_HOG_features(indx2,:),HOG_feature(indx1,:));
            sim_values(indx1,indx2,2) = pdist2(past_colrhist_feature(indx2,:),colrhist_feature(indx1,:));
%             sim_values(indx1,indx2,3) =  sqrt(sum((past_centroid_feat(indx2,:) - centroid_feat(indx2,:)) .^ 2));
%             sim_values(indx1,indx2,3) = corrcoef(past_hus_feature(indx2,:),hus_feature(indx1,:));
             end
         end
         sim_values(:,:,1) = (sim_values(:,:,1))./max((sim_values(:,:,1)));
         sim_values(:,:,2) = (sim_values(:,:,2))./max((sim_values(:,:,2)));
      
         
         [min_1] = min(sim_values(:,:,1),[],2);
         [min_2] = min(sim_values(:,:,2),[],2);       
%          [min_3] = min(sim_values(:,:,3),[],2);
         [~,min_indx1] = min(sim_values(:,:,1),[],2);
         [~,min_indx2] = min(sim_values(:,:,2),[],2);       
%          [~,min_indx3] = min(sim_values(:,:,3),[],2);
   
         
        X(l_indx,:)=  [min_1(1)';min_2(1)';];
%         for indx = 1:length(sim_values(:,1,1))
%             X_ip = [sim_values(indx,:,1);sim_values(indx,:,2);sim_values(indx,:,3)];
        
%         tracks_i = find(labels);
%         new_track = min_indx2(tracks_i);
      
          
%         end
%         track_i = find(labels);
         clear past_HOG_features;
         clear past_colrhist_feature;
         clear sim_values;
         clear score;
         clear labels;
    end
     clear BB_box_num
    
    if(l_indx > 1)
    past_HOG_features = HOG_feature;
    past_colrhist_feature = colrhist_feature;
    past_centroid_feat = centroid_feat;
    clear HOG_feature;
    clear colrhist_feature;
%     past_hus_feature = hus_feature;
    else
        past_HOG_features = zeros(1,32);
        past_colrhist_feature = zeros(1,24);
        past_centroid_feat = zeros(1,2);
    end
    n_indx = n_indx + 1;
end
[labels,score] = predict(Mdl_ada,X');
 save(['ada_boost1_mdl'],'Mdl_ada');
