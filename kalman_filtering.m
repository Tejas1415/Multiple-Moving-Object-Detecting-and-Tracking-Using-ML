function [x,P] = kalman_filtering(x,P,z)
H = eye(4);
Q = 0.5*eye(4);
F = eye(4);
R = 5*eye(4);
[x,P] = kalman_predict(x,P,F,Q);
[x,P] = kalman_update(x,P,z,H,R);
end

function [x,P] = kalman_predict(x,P,F,Q)
    x = F*x; %predicted state
    P = F*P*F' + Q; %predicted estimate covariance
end

function [x,P] = kalman_update(x,P,z,H,R)
    y = z - H*x; %measurement error/innovation
    S = H*P*H' + R; %measurement error/innovation covariance
    K = P*H'/S; %optimal Kalman gain
    x = x + K*y; %updated state estimate
    P = (eye(size(x,1)) - K*H)*P; %updated estimate covariance
end

