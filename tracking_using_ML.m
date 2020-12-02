%% training a adaboost classifier
close all; clear; clc;
% Reading annotations file

fileID = fopen('MVI_20012.xml','r');
A = fscanf(fileID,'%c');
idcs = strfind(A,['num=']);
lost_cnt = 5;
detect_cnt = 2;
l_indx = 1;
full_filename ='highway.avi';
vidObj = VideoReader(full_filename);
display_rate = 2;
% Canny edge detection
T1 = 0.2;
T2 = 0.05;
%Blob parameters and States%
max_blob_cnt = 8;
morph_param = 12;
blob_state_struct = struct('BB_Box_pst',zeros(max_blob_cnt,4),...
    'Centroid_pst',zeros(max_blob_cnt,2),'Area_pst',...
    zeros(max_blob_cnt,1),'BB_Box_cur',zeros(max_blob_cnt,4),...
    'Centroid_cur',zeros(max_blob_cnt,2),'Area_cur',...
    zeros(max_blob_cnt,1),'BB_Box_new',zeros(max_blob_cnt,4),...
    'Centroid_new',zeros(max_blob_cnt,2),'Area_new',zeros(max_blob_cnt,1),...
    'lost_th',lost_cnt);
blob_state = repmat(blob_state_struct, 1, 1);
blob_cnt_struct = struct('detect_cnt',0,'lost_cnt',0);
blob_data = repmat(blob_cnt_struct, max_blob_cnt, 1);
%Kalman Filter State%
kalman_struct = struct('X',zeros(4,1),'P',eye(4));
kalman_data = repmat(kalman_struct, max_blob_cnt, 1 );
%Initial parameters%
img_file_name = ['img','0000',num2str(1),'.jpg'];
folder = 'MVI_20012';
fullname = fullfile(folder,img_file_name);
%Previousframe = imread(fullname);
Previousframe = readFrame(vidObj);
frame = 1;
%Display Parameters%
BoundingBox_N = int32(zeros(max_blob_cnt,4));
blue = [255,20,147];
yellow = [255,255,0];
%Debug Parameters%
debug = 1;

     load ada_boost1_mdl.mat;
        Mdl_ada = Mdl_ada;
% Main Functionality
while hasFrame(vidObj)%for l_indx = 1:length(idcs)
    %Reading the current frame
    if(l_indx < 10)
        str1 =  '0000';
    elseif(l_indx < 100)
        str1 =  '000';
    else
        str1 = '00';
    end
    img_file_name = ['img',str1,num2str(l_indx),'.jpg'];
    
    fullname = fullfile(folder,img_file_name);
%     Currentframe = imread(fullname);
	Currentframe = readFrame(vidObj);
    %Background Subtraction
    subframe = Currentframe - Previousframe;
    %Canny Edge Detection
    canny_gradient = canny_edge_detection(subframe, T1, T2);
    %Preparing Input for Blob Analysis
    Blob_ip = imclose(canny_gradient, strel('rectangle',...
        [morph_param,morph_param]));
    %Blob Detection using Blob Analysis Vision Tool Box
    detected_blobs = (logical(Blob_ip));


        Blobdetection = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'MaximumCount',max_blob_cnt,'MinimumBlobArea',300,'Connectivity',4);



        [Area,Centroid,BoundingBox] = step(Blobdetection,detected_blobs);

    %     tic
    %     fprintf('Region Props');
%     [stats] = regionprops(detected_blobs, 'BoundingBox','Centroid','Area');
    %     toc
    %Initializing the states before reordering
%     idx = find([stats.Area] > 300);
%     stats = stats(idx);
%     NUM_OF_BLOBS = length(stats);
    NUM_OF_BLOBS = length(Area);
%     BoundingBox = reshape([stats.BoundingBox],[4,NUM_OF_BLOBS])';
%     Centroid = reshape([stats.Centroid],[2,NUM_OF_BLOBS])';
%     Area = reshape([stats.Area],[1,NUM_OF_BLOBS])';
    blob_state.BB_Box_cur = BoundingBox;
    blob_state.Centroid_cur = Centroid;
    blob_state.Area_cur = Area;
    % Reordering Detected Blobs
    if ((frame > 2) && (NUM_OF_BLOBS > 0))
        for indx = 1:NUM_OF_BLOBS
            crop_img = imcrop(Currentframe,BoundingBox(indx,:));
            cell_num1 = (length(crop_img(:,1,1)));
            cell_num2 = (length(crop_img(1,:,1)));
            cell_num1 = floor(cell_num1/2);
            cell_num2 = floor(cell_num2/2);
            tmp = [extractHOGFeatures(crop_img,'numbins',8,'cellsize',[cell_num1,cell_num2])];
            HOG_feature(indx, :) = tmp;
            HOG_feature(indx, :) = (HOG_feature(indx, :) );
            colrhist_feature(indx,:) = [imhist(crop_img(:,:,1),8);...
                imhist(crop_img(:,:,2), 8);imhist(crop_img(:,:,2), 8)]';
            centroid_feat(indx,:) = [BoundingBox(indx,1)+...
                (BoundingBox(indx,1)/2) ,BoundingBox(indx,3)+(BoundingBox(indx,4)/2)];
%             colrhist_feature(indx,:) = (colrhist_feature(indx, :) );
            %             hus_feature(indx, :) =  hus_invariance(crop_img);
        end
        NUM_OF_PAST_BLOBS = length(past_colrhist_feature(:,1)) ;
        for indx1 = 1:NUM_OF_BLOBS
            for indx2 = 1:NUM_OF_PAST_BLOBS
                sim_values(indx1,indx2,1) = pdist2(past_HOG_features(indx2,:),HOG_feature(indx1,:));
                sim_values(indx1,indx2,2) = pdist2(past_colrhist_feature(indx2,:),colrhist_feature(indx1,:));
                sim_values(indx1,indx2,3) =  sqrt(sum((past_centroid_feat(indx2,:) - centroid_feat(indx2,:)) .^ 2));
                %             sim_values(indx1,indx2,3) = corrcoef(past_hus_feature(indx2,:),hus_feature(indx1,:));
            end
        end
       
          sim_values(:,:,1) = (sim_values(:,:,1))./max(sim_values(:,:,1));
         sim_values(:,:,2) = (sim_values(:,:,2))./max(sim_values(:,:,2));

        %Reordering Blobs
%         
%         [~,m_indx1] = min(sim_values(:,:,1),[],2);
%         [~,m_indx2] = min(sim_values(:,:,2),[],2);
%         %[~,m_indx3] = min(sim_values(:,:,2),[],2);
%         [m_val1] = min(sim_values(:,:,1),[],2);
%         [m_val2] = min(sim_values(:,:,2),[],2);
%         %[m_val3] = min(sim_values(:,:,2),[],2);
         for indx1 = 1:NUM_OF_BLOBS
             for indx2 = 1:NUM_OF_PAST_BLOBS
                [label,score] = predict(Mdl_ada,[sim_values(indx1,indx2,1);sim_values(indx1,indx2,2)]');
                score_ip(indx1,indx2) = score(1);
             end
             
         end
        
        if(NUM_OF_BLOBS == 1)
            label = 1;
            score = [-50, 50];
        end
        tracks_i = find(label);

        new_track = 1;%m_indx2(tracks_i);
        
        clear past_HOG_features;
        clear past_colrhist_feature;
        clear sim_values;

         %Reordering Blobs based on score
        [nw_order,blob_state,blob_data] = blob_state_reorder...
            (blob_state,blob_data, score_ip);
       
        %Preparing Inputs for kalman Filter State
        [kalman_data,blob_data,blob_state] = Kalman_state_reorder...
            (nw_order,kalman_data,...
            blob_data,blob_state);     
        BB_Box_new = blob_state.BB_Box_new;
        NUM_OF_BLOBS = length(BB_Box_new(:,4));   
        %Display Parameters
        disp_cnt = 1;
        BB_dsply = int32(zeros(NUM_OF_BLOBS,4));
        BB_dsply_N = int32(zeros(NUM_OF_BLOBS,4));
        %Kalman Filtering each multiple moving object
        for k = 1:NUM_OF_BLOBS
            % Updating Kalman State for each Box
            Z = double(BB_Box_new(k,:))';       
            [kalman_data(k).X,kalman_data(k).P] = kalman_filtering...
                (kalman_data(k).X,kalman_data(k).P,Z);
            
            BoundingBox_N(k,:) =int32(kalman_data(k).X);
            % Displaying each frame for debugging purpose
            if(blob_data(k).detect_cnt > detect_cnt)
                BB_dsply(disp_cnt,:) = int32(BB_Box_new(k,:));
                BB_dsply_N(disp_cnt,:) = int32(BoundingBox_N(k,:));
                disp_cnt = disp_cnt+1;
            end
        end
        %Displaying the output
        shapeInserter = vision.ShapeInserter('BorderColor','Custom',...
            'CustomBorderColor',blue);
        out1    = step(shapeInserter, Currentframe, BB_dsply);
        % Displaying Boundary Box of Kalman Predicted Objects in Video
        shapeInserter = vision.ShapeInserter('BorderColor','Custom',...
            'CustomBorderColor',yellow);
        out2    = step(shapeInserter, out1, BB_dsply_N);    
        if(mod(frame,display_rate)==0)
            imshow(out2);
        end
    end
    %Updating Parameters for Next Iteration
    Previousframe = Currentframe;
    if(frame > 2)
        if(NUM_OF_BLOBS == 0)
            past_HOG_features = zeros(1,32);
        past_colrhist_feature = zeros(1,24);
        past_centroid_feat = int32(zeros(1,2));
        pst_track = 0;
        else
        past_HOG_features = HOG_feature;
        pst_track = new_track;
        past_colrhist_feature = colrhist_feature;
        clear HOG_feature;
        clear colrhist_feature;
        clear score;
        past_centroid_feat = centroid_feat;
        end
        %     past_hus_feature = hus_feature;
    else
        past_HOG_features = zeros(1,32);
        past_colrhist_feature = zeros(1,24);
        past_centroid_feat = int32(zeros(1,2));
        pst_track = 0;
    end
    
    if (frame > 2)
        blob_state.BB_Box_pst = blob_state.BB_Box_new;
        blob_state.Centroid_pst = blob_state.Centroid_new;
        blob_state.Area_pst = blob_state.Area_new;
    else
        blob_state.BB_Box_pst = blob_state.BB_Box_cur;
        blob_state.Centroid_pst = blob_state.Centroid_cur;
        blob_state.Area_pst = blob_state.Area_cur;
    end
    frame = frame +1;
    if(debug)
        if (frame == 86)
            frame = frame;
        end
    end
end







