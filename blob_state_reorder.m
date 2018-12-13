
function [nw_order,blob_state,blob_data] = blob_state_reorder...
    (blob_state,blob_data, score)
%Loading state variables to local variables
cur_cnt = length(blob_state.Area_cur);
pst_cnt = length(blob_state.Area_pst);
BB_Box_cur = blob_state.BB_Box_cur;
BB_Box_pst = blob_state.BB_Box_pst;
C_pst = blob_state.Centroid_pst;
C_cur = blob_state.Centroid_cur;
A_pst = blob_state.Area_pst;
A_cur = blob_state.Area_cur;
%Declaring new variables 
nw_cnt = max(cur_cnt, pst_cnt);
C_new = zeros(nw_cnt,2);
A_new = zeros(1,nw_cnt);
BB_Box_new = zeros(nw_cnt,4);
nw_order = zeros(nw_cnt,1);
dist_parm = zeros(pst_cnt,1);
D_cur = zeros(cur_cnt,2);
D_pst = zeros(pst_cnt,2);
%Initializations
k = 1;
Th1 = 0;
d_cnt = 1;
%Reording the blobs based on min distance between frames
while(k <= pst_cnt) 
    d_prm = 100*ones(1,cur_cnt);
    for i = 1:cur_cnt 
        if(D_cur(i,1) == 0)
            %d_prm(i) = sum(abs(C_cur(i,:)-C_pst(k,:)));
			d_prm(i) = score(i,k);
        end
    end  
    [min_d, min_i] = min(d_prm);
    dist_parm(k) = min_d;
    if(min_d < Th1)
        C_new(d_cnt,:) = C_cur(min_i,:);
        A_new(d_cnt) = A_cur(min_i);
        BB_Box_new(d_cnt,:) = BB_Box_cur(min_i,:);
        nw_order(d_cnt) = k;
        D_cur(min_i,:) = [-1,-1];
        D_pst(k,:)= [-1,-1];% as a marker to know if the value is set
        %Updating Detect Counter
        blob_data(k).detect_cnt = blob_data(k).detect_cnt+1;
        d_cnt= d_cnt+1;
    end
    k = k+1;
end

 m = 1;
if(pst_cnt > cur_cnt)
    for i = 1:pst_cnt
        if(D_pst(i,1) == 0)
            nw_order(d_cnt-1+m) = i;
            C_new(d_cnt-1+m,:) = C_pst(i,:);
            A_new(d_cnt-1+m) = A_pst(i);
            BB_Box_new(d_cnt-1+m,:) = BB_Box_pst(i,:);
            %Updating Lost Counter
            blob_data(i).lost_cnt = blob_data(i).lost_cnt+1;
            m=m+1;
        end
    end
elseif(pst_cnt < cur_cnt)
    for i = 1:cur_cnt
        if(D_cur(i,1) == 0)
            nw_order(d_cnt-1+m) = d_cnt-1+m;
            C_new(d_cnt-1+m,:) = C_cur(i,:);
            A_new(d_cnt-1+m) = A_cur(i);
            BB_Box_new(d_cnt-1+m,:) = BB_Box_cur(i,:);
            m=m+1;
        end
    end
else
     for i = 1:cur_cnt
        if(D_cur(i,1) == 0)
            nw_order(d_cnt-1+m) = i;
            C_new(d_cnt-1+m,:) = C_cur(i,:);
            A_new(d_cnt-1+m) = A_cur(i);
            BB_Box_new(d_cnt-1+m,:) = BB_Box_cur(i,:);
            %Updating Lost Counter
            blob_data(i).lost_cnt = blob_data(i).lost_cnt+1;
            m=m+1;
        end
    end
end
% Loading new found blob locations to the filter state
blob_state.BB_Box_new =BB_Box_new; 
blob_state.Centroid_new =C_new; 
blob_state.Area_new =A_new; 
