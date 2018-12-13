
function [kalman_data,blob_data,blob_state] = Kalman_state_reorder...
    (nw_order,kalman_data,blob_data,blob_state)
%variable initialization
tmp_kalman_data = kalman_data;
tmp_blob_data = blob_data;
max_cnt = length(blob_state.Area_new);
pst_cnt = length(blob_state.Area_pst);
A_new = blob_state.Area_new;
C_new = blob_state.Centroid_new;
BB_new = blob_state.BB_Box_new;
%Rearranging the filter orders and initializing new states
for i = 1:max_cnt
    kalman_data(i) = tmp_kalman_data(nw_order(i));
    blob_data(i) = tmp_blob_data(nw_order(i));
    if(blob_data(i).detect_cnt == 1)%Initializing the states
        kalman_data(i).X = blob_state.BB_Box_new(i,:)';
    end   
end

%Deleting the lost filters
m = 0;
for i = 1:max_cnt
    if(blob_data(i).lost_cnt > blob_state.lost_th)%Deleting the states      
        nw_cnt = length(A_new);
        pos = nw_cnt - (max_cnt-i);
        if(pos < nw_cnt)
            A_new = [A_new(1:pos-1),A_new(pos+1:nw_cnt)];
            C_new = [C_new(1:pos-1,:);C_new(pos+1:nw_cnt,:)];
            BB_new = [BB_new(1:pos-1,:);BB_new(pos+1:nw_cnt,:)];
        else
            A_new = A_new(1:pos-1);
            C_new = C_new(1:pos-1,:);
            BB_new = BB_new(1:pos-1,:);
        end
        skip=1;
    else
        m = m+1;
        skip = 0;
    end
    if(~skip)
        kalman_data(m) = kalman_data(i);
        blob_data(m) = blob_data(i);
    end
end
% Resetting the deleted states to zeros
if (m < max_cnt)
    resest_cnt = m+1;
    for i = resest_cnt:max_cnt
        kalman_data(i).X = zeros(4,1);
        kalman_data(i).P = eye(4);
        blob_data(i).lost_cnt = 0;
        blob_data(i).detect_cnt = 0;
    end
end
% assigning them back to the state
blob_state.Area_new = A_new;
blob_state.Centroid_new = C_new;
blob_state.BB_Box_new = BB_new;