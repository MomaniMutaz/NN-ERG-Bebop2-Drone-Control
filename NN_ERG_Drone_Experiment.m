clear all
close all
clc

% This script controls a Parrot Bebop 2 drone in real-time using motion capture feedback and a neural-network-based Explicit Reference Governor (NN ERG) for adaptive trajectory tracking and safe obstacle avoidance.

% Declare global variables used throughout the script
% These include communication interfaces, trajectory points, control flags, system state, and model parameters
global SIASclient    FLAG
global goalDesired Error CommandInput SS SS2 commandArray_Idf Cont GoalPt CI
global deltaT inc drone_mode ContGoal xyDesired vertDesired yawDesired DESIREDPOINT
global p state state2 xyDesired vertDesired rollDesired pitchDesired yawDesired DESIRED
global SIASclient Error CommandInput ContGoal Time Time1
global goalDesired SS SS2 commandArray_Idf CI kYaw kD_Yaw
global deltaT drone_mode Cont GoalPt DESIRED1  v kappa
global p state state2 inc goalDesired deltaT  xyDesired vertDesired rollDesired pitchDesired yawDesired c d DESIREDPOINT  zeta delta smoothing_factor lambda  tolerance dt_v A B K P m_1 m_2 G mu f1 f2 net tolerance c01 c02 radius a b k1 k2 Beta gamma






% Initialize and connect to the NatNet client (motion capture system)
SIASclient = natnet;
SIASclient.connect;
pause(2)

fprintf('\n\nConnecting to Drone...\n')
p = parrot();
fprintf('Connected to %s\n', p.ID)
fprintf('Battery Level is %d%%\n', p.BatteryLevel)

% Take off and hover
takeoff(p);
pause(1)





% the desired reference
goalDesired = [0 0; -1.5 1.8 ;1.5 1.5;0 0;0 0;0 0];



zeta=0.5;% the distance from the constraint for which the repulsive force effect begins
delta=0.2; % the distance from the constraint from where the tightned constraint exists, and the repulsive force ...
% factor becomes equal to 1 



smoothing_factor=0.01; 

lambda=0.01; % smoothing factor 2









 

dt_v=0.083;





%  state space model 

Kx1=-0.052701624162968;
Kx2=-5.477869892924388;
Ky1=-0.018687914020970;
Ky2=-7.060834512263481;
Kz1=-1.787280741456883;
Kz2=-1.738212699869965;

A=[0   1   0   0   0   0
   0  Kx1  0   0   0   0
   0   0   0   1   0   0
   0   0   0   Ky1 0   0
   0   0   0   0   0   1
   0   0   0   0   0  Kz1];

B=[0   0   0
  Kx2  0   0
   0   0   0
   0  Ky2  0
   0   0   0
   0   0  Kz2];





% Proportional Gain Values
kYaw =-1;
kPitch = -0.08; 
kRoll = -0.08; 
kVert = -1.7;
% Derivative Gain Values
kD_Pitch = -0.06; 
kD_Roll = -0.06; 
kD_Yaw =-0.8;
kD_Vert = -0.05;

%state feedback gain matrix
K=[kPitch kD_Pitch 0 0 0 0;
    0 0 kRoll kD_Roll 0 0;
    0 0 0 0 kVert kD_Vert];



A_bar=A-B*K;

I = eye(size(A_bar,2)); % pos.def matrix
P = lyap(A_bar, 2*I);% % P matrix for the Lyapunov function






% minimum and maximum eigenvalues of the P matrix
m_1=min(eig(P));
m_2=max(eig(P));

%feedforward gain
G=inv((-inv(A-B*K)*B)'*(-inv(A-B*K)*B))*(-inv(A-B*K)*B)'*[1 0 0;0 0 0;0 1 0;0 0 0;0 0 1;0 0 0]; 

%lipschitz
mu=norm(-inv(A-B*K)*B*G);


x=-2:0.01:2;

% defining the wall constraints f1 & f2
b = 0; % go up or down
a = 0;
k1 = -4;
k2 = 4;
Beta = 1/((x(1))^2);



f1 =@(x) ((1/2*(a-k1))*Beta*(x-b).^2+(1/2)*(a+k1));

f2 =@(x) ((1/2*(a-k2))*Beta*(x-b).^2+(1/2)*(a+k2));



% Load the trained network
loaded_net = load('trained_network_drone.mat');
net = loaded_net.net;


tolerance =     0.531382759839911;

c01=[-0.5;-0.5]; % the center of the first circle
c02=[0.5;0.5]; % the center of the second circle
radius=0.3;    % the radius for the circle















Error=zeros(4,1);
CommandInput=zeros(4,1);
inc=1;
Cont=1;
ContGoal=0;
GoalPt=1;


while(1>0)

tic

    Time(inc,1)=double(SIASclient.getFrame.fTimestamp);





    Position=double([SIASclient.getFrame.RigidBodies(1).x;SIASclient.getFrame.RigidBodies(1).y;SIASclient.getFrame.RigidBodies(1).z]);
    q=quaternion( SIASclient.getFrame.RigidBodies(1).qw, SIASclient.getFrame.RigidBodies(1).qx, SIASclient.getFrame.RigidBodies(1).qy, SIASclient.getFrame.RigidBodies(1).qz );
    eulerAngles=quat2eul(q,'xyz')*180/pi;
    Angle=[eulerAngles(1);eulerAngles(2);eulerAngles(3)];
    state=[Position;Angle];
    SS(:,inc)=state;
    [errorArray]=ControlCommand;

    if FLAG(inc)==1
        SS(:,inc)=state;
        CI(:,inc)=commandArray_Idf;
        

        DESIRED(:,inc)=[goalDesired(1:2,GoalPt);goalDesired(3,GoalPt); goalDesired(6,GoalPt)];
        Error(:,inc)=errorArray;
        inc=inc+1;

    end
    
end


function [errorArray]=ControlCommand

global p state  inc FLAG SS CI Error kappa v commandArray_Idf  GoalPt ContGoal goalDesired  Time DESIRED  zeta delta smoothing_factor lambda  kYaw kD_Yaw dt_v Time Time1 A B K P m_1 m_2 G mu c d f1 f2  net tolerance c01 c02 radius a b k1 k2 Beta gamma


%% Define desired tolerances and gains






xyDesired(1:2,1) = goalDesired(1:2,GoalPt);
vertDesired = goalDesired(3,GoalPt);
yawDesired= goalDesired(6,GoalPt);



r=xyDesired;
height=vertDesired;



positionActual = state(1:2);
vertActual = state(3);
yawActual = deg2rad(state(6)); 


x=positionActual(1);
y=positionActual(2);
z=vertActual;





% Compute the errors
% Yaw Error
if inc==1
    dt=1/120;
else
    dt=Time(inc,1)-Time(inc-1,1);
end

if dt < eps
    dt = 1/120; % Average calculated time step
end

dt_v=10/120;


if inc == 1

 old_yaw_Error = 0; 

 x_dot=0;
 y_dot=0;
 z_dot=0;

X(:,inc)=[x; x_dot; y; y_dot; z; z_dot];



v0=[X(1,inc);X(3,inc)];


gamma1 = net(v0)-tolerance; % compute the threshold for the wall constraints using neural network, % other options like dlnetwork with `predict` may offer faster execution

% compute the threshold for the control input constraints
gamma2 = ((K(1,:)*[v0(1);0;v0(2);0;height;0]-G(1,:)*[v0(1);v0(2);height]+0.06)^2)/(K(1,:)*inv(P)*K(1,:)');

gamma3 = ((-K(1,:)*[v0(1);0;v0(2);0;height;0]+G(1,:)*[v0(1);v0(2);height]+0.06)^2)/(K(1,:)*inv(P)*K(1,:)');

gamma4 = ((K(2,:)*[v0(1);0;v0(2);0;height;0]-G(2,:)*[v0(1);v0(2);height]+0.06)^2)/(K(2,:)*inv(P)*K(2,:)');

gamma5 = ((-K(2,:)*[v0(1);0;v0(2);0;height;0]+G(2,:)*[v0(1);v0(2);height]+0.06)^2)/(K(2,:)*inv(P)*K(2,:)');

% compute the threshold for the first obstacle constraint
xv_telda1 = -radius*(([c01(1);0;c01(2);0;height;0] - [v0(1);0;v0(2);0;height;0])/norm([c01(1);0;c01(2);0;height;0] - [v0(1);0;v0(2);0;height;0]))+ [c01(1);0;c01(2);0;height;0];
xv_bar = [v0(1);0;v0(2);0;height;0];

partial_O1 = [2*(xv_telda1(1)-c01(1));  0; 2*(xv_telda1(3)-c01(2)); 0; 0; 0];

gamma6 = ((partial_O1'*(xv_bar-xv_telda1))^2)/(partial_O1'*inv(P)*partial_O1); % the first tree

% compute the threshold for the second obstacle constraint
xv_telda2 = -radius*(([c02(1);0;c02(2);0;height;0] - [v0(1);0;v0(2);0;height;0])/norm([c02(1);0;c02(2);0;height;0] - [v0(1);0;v0(2);0;height;0]))+ [c02(1);0;c02(2);0;height;0];
xv_bar = [v0(1);0;v0(2);0;height;0];

partial_O2 = [2*(xv_telda2(1)-c02(1));  0; 2*(xv_telda2(3)-c02(2)); 0; 0; 0];

gamma7 = ((partial_O2'*(xv_bar-xv_telda2))^2)/(partial_O2'*inv(P)*partial_O2);


% take the minimum among the thresholds
gamma(inc)= min([gamma1,gamma2, gamma3, gamma4, gamma5, gamma6, gamma7]);












% Compute DSM
Lyapunov=double((X(:,inc)-[v0(1);0;v0(2);0;height;0])'*P*(X(:,inc)-[v0(1);0;v0(2);0;height;0]));

DSM=max(gamma(inc)-Lyapunov,0);

%Compute NF rho_a is the attraction term and rho_r represent the
%repulsive term
rho_a=(r-v0)/max(norm(r-v0),smoothing_factor);
rho_a = round(rho_a,2);


c1=v0(2)-f1(v0(1));  
rho_r1= max(0, (zeta - c1) / (zeta - delta)) * [(-4*(a-k1)*(v0(1)-b))/max(norm(-4*(a-k1)*(v0(1)-b)),0.01); 1/norm(1)]  ;

c2=f2(v0(1))-v0(2);  

rho_r2= max(0, (zeta - c2) / (zeta - delta)) * [(Beta*(a-k2)*(v0(1)-b))/max(norm(Beta*(a-k2)*(v0(1)-b)),0.01);-1/norm(-1)];


phi1=sqrt((v0(1)-c01(1))^2 + (v0(2)-c01(2))^2);

c3=phi1 - radius;

rho_r3=max( (zeta - c3) /(zeta - delta)  , 0) * [(v0(1)-c01(1))/phi1; (v0(2)-c01(2))/phi1];

phi2=sqrt((v0(1)-c02(1))^2 + (v0(2)-c02(2))^2);

c4=phi2 - radius;

rho_r4=max( (zeta - c4) /(zeta - delta)  , 0) * [(v0(1)-c02(1))/phi2; (v0(2)-c02(2))/phi2];




rho_r=rho_r1+rho_r2+rho_r3+rho_r4;


Attraction_Field=rho_a+rho_r;
g=DSM*Attraction_Field;




% the equations for computing the adaptive kappa value
vartheta=min(c1, min(c2,min(c3,c4)))-delta; % the distance of the current equilibruim point from the virtual constraint

kappa(inc)=max(((sqrt(m_1)/(sqrt(m_1)+sqrt(m_2)))*vartheta-(sqrt(m_2)/(sqrt(m_1)+sqrt(m_2)))*(norm(X(:,inc)-[v0(1);0;v0(2);0;height;0])))/(mu*dt_v*max(norm(g),lambda)),0); %dynamic kappa

vdot=kappa(inc)*g;

v(:,inc)=v0+vdot*dt_v; % the applied reference update
u(:,inc)=-K*X(:,inc)+G*[v(1,inc);v(2,inc);height]; % compute the control command





    

else % if inc is not equal to 1


old_yaw_Error=Error(3,inc-1);



x_dot=(SS(1,inc)-SS(1,inc-1))/dt;
y_dot=(SS(2,inc)-SS(2,inc-1))/dt;
z_dot=(SS(3,inc)-SS(3,inc-1))/dt;

X(:,inc)=[x; x_dot; y; y_dot; z; z_dot];



if rem(Time(inc,1),dt_v)==0

gamma1 = net(v(:,inc-1))-tolerance; %% compute the threshold for the wall constraints using neural network, % other options like dlnetwork with `predict` may offer faster execution

% compute the threshold for the control input constraints
gamma2 = ((K(1,:)*[v(1,inc-1);0;v(2,inc-1);0;height;0]-G(1,:)*[v(1,inc-1);v(2,inc-1);height]+0.06)^2)/(K(1,:)*inv(P)*K(1,:)');

gamma3 = ((-K(1,:)*[v(1,inc-1);0;v(2,inc-1);0;height;0]+G(1,:)*[v(1,inc-1);v(2,inc-1);height]+0.06)^2)/(K(1,:)*inv(P)*K(1,:)');

gamma4 = ((K(2,:)*[v(1,inc-1);0;v(2,inc-1);0;height;0]-G(2,:)*[v(1,inc-1);v(2,inc-1);height]+0.06)^2)/(K(2,:)*inv(P)*K(2,:)');

gamma5 = ((-K(2,:)*[v(1,inc-1);0;v(2,inc-1);0;height;0]+G(2,:)*[v(1,inc-1);v(2,inc-1);height]+0.06)^2)/(K(2,:)*inv(P)*K(2,:)');

% compute the threshold for the first obstacle constraint
xv_telda1 = -radius*(([c01(1);0;c01(2);0;height;0] - [v(1,inc-1);0;v(2,inc-1);0;height;0])/norm([c01(1);0;c01(2);0;height;0] - [v(1,inc-1);0;v(2,inc-1);0;height;0]))+ [c01(1);0;c01(2);0;height;0];

xv_bar = [v(1,inc-1);0;v(2,inc-1);0;height;0];

partial_O1 = [2*(xv_telda1(1)-c01(1));  0; 2*(xv_telda1(3)-c01(2)); 0; 0; 0];

gamma6 = ((partial_O1'*(xv_bar-xv_telda1))^2)/(partial_O1'*inv(P)*partial_O1);

% compute the threshold for the second obstacle constraint
xv_telda2 = -radius*(([c02(1);0;c02(2);0;height;0] - [v(1,inc-1);0;v(2,inc-1);0;height;0])/norm([c02(1);0;c02(2);0;height;0] - [v(1,inc-1);0;v(2,inc-1);0;height;0]))+ [c02(1);0;c02(2);0;height;0];

xv_bar = [v(1,inc-1);0;v(2,inc-1);0;height;0];

partial_O2 = [2*(xv_telda2(1)-c02(1));  0; 2*(xv_telda2(3)-c02(2)); 0; 0; 0];

gamma7 = ((partial_O2'*(xv_bar-xv_telda2))^2)/(partial_O2'*inv(P)*partial_O2);


% take the minimum among the thresholds
gamma(inc) = min([gamma1,gamma2, gamma3, gamma4, gamma5, gamma6, gamma7]);



% compute the DSM
Lyapunov=double((X(:,inc)-[v(1,inc-1);0;v(2,inc-1);0;height;0])'*P*(X(:,inc)-[v(1,inc-1);0;v(2,inc-1);0;height;0]));

DSM=max(gamma(inc)-Lyapunov,0);

%Compute NF rho_a is the attraction term and rho_r represent the
%repulsive term
rho_a=(r-v(:,inc-1))/max(norm(r-v(:,inc-1)),smoothing_factor);
rho_a = round(rho_a,2);
if norm(r-v(:,inc-1))<=0.02
    rho_a=[0;0];
end

c1=v(2,inc-1)-f1(v(1,inc-1));  %vy-f1(vx)  0-Beta*(a-k1)*(v0(1)-b)

rho_r1= max(0, (zeta - c1) / (zeta - delta)) * [(-Beta*(a-k1)*(v(1,inc-1)-b))/max(norm(-Beta*(a-k1)*(v(1,inc-1)-b)),0.01); 1/norm(1)]  ;

c2=f2(v(1,inc-1))-v(2,inc-1);  %f2(vx)-vy  Beta*(a-k2)*(v0(1)-b)

rho_r2= max(0, (zeta - c2) / (zeta - delta)) * [(Beta*(a-k2)*(v(1,inc-1)-b))/max(norm(Beta*(a-k2)*(v(1,inc-1)-b)),0.01);-1/norm(-1)];

phi1=sqrt((v(1,inc-1)-c01(1))^2 + (v(2,inc-1)-c01(2))^2);

c3=phi1 - radius;

rho_r3=max( (zeta - c3) /(zeta - delta)  , 0) * [(v(1,inc-1)-c01(1))/phi1; (v(2,inc-1)-c01(2))/phi1];


phi2=sqrt((v(1,inc-1)-c02(1))^2 + (v(2,inc-1)-c02(2))^2);

c4=phi2 - radius;

rho_r4=max( (zeta - c4) /(zeta - delta)  , 0) * [(v(1,inc-1)-c02(1))/phi2; (v(2,inc-1)-c02(2))/phi2];



rho_r=rho_r1+rho_r2+rho_r3+rho_r4;

Attraction_Field=rho_a+rho_r;

g=DSM*Attraction_Field;




% the equations for computing the adaptive kappa value
vartheta=min(c1, min(c2,min(c3,c4)))-delta; % the distance of the current equilibruim point from the virtual constraint

kappa(inc)=max(((sqrt(m_1)/(sqrt(m_1)+sqrt(m_2)))*vartheta-(sqrt(m_2)/(sqrt(m_1)+sqrt(m_2)))*(norm(X(:,inc)-[v(1,inc-1);0;v(2,inc-1);0;height;0])))/(mu*dt_v*max(norm(g),lambda)),0); %dynamic kappa



vdot=kappa(inc)*g;

v(:,inc)=v(:,inc-1)+vdot*dt_v; % the applied reference update
u(:,inc)=-K*X(:,inc)+G*[v(1,inc);v(2,inc);height]; % compute the control command



else % these ones were for updating v in discrete time

        v(:,inc)=v(:,inc-1);
        kappa(inc)=kappa(inc-1);
        gamma(inc)=gamma(inc-1);
        u(:,inc)=-K*X(:,inc)+G*[v(1,inc);v(2,inc);height];





end

end

% to rotate X,Y from world frame to robot frame
Tw2r = [cos(yawActual), sin(yawActual); -sin(yawActual), cos(yawActual)];




yawe1=(yawDesired - yawActual);
yawError = wrapToPi(yawe1);
yawD_Error = (yawError-old_yaw_Error)/dt;

% compute the yaw commands
yawCmd = kYaw*yawError+kD_Yaw*yawD_Error;

if abs(yawCmd) > 3.4
    yawCmd = sign(yawCmd)*3.4;
end



% Position Error
xyError = xyDesired(1:2,1) - positionActual;
% xyD_Error = (xyError - old_xy_Error)/dt;

% compute the pitch commands
% roll_pitch_cmd = (Tw2r)*xyError; %error in robot frame

% pitchCmd = kPitch*roll_pitch_cmd(1) + kD_Pitch*xyD_Error(1);

% la
pitchCmd = 1*u(2,inc);


if abs(pitchCmd) > 0.05 % limitations of Parrot Drone
    pitchCmd = sign(pitchCmd)*0.05;
end

% compute the roll commands
% rollCmd = kRoll*roll_pitch_cmd(2)+ kD_Roll*xyD_Error(2);

% rollCmd = u(2,inc);
rollCmd = -1*u(1,inc);



if abs(rollCmd) > 0.05 % limitations of Parrot Drone
    rollCmd = sign(rollCmd)*0.05;
end


% Altitude Error
vertError = vertDesired - vertActual;
% vertD_Error = (vertError - old_vert_Error)/dt;

% vertCmd = kVert*vertError+kD_Vert*vertD_Error;

vertCmd = u(3,inc);

if vertCmd < -0.6
    vertCmd = -0.6;
elseif vertCmd > 0.6
    vertCmd = 0.6;
end


TotalError = norm([xyError; yawError; vertError]);

% store data for post analysis
errorArray = [xyError; yawError; vertError];
% commandArray= [pitchCmd; rollCmd; yawCmd; vertCmd];
% TotalError = norm(totalError);

if inc>1
    if Time(inc,1)-Time(inc-1,1)>eps
        commandArray_Idf= [pitchCmd; rollCmd; yawCmd; vertCmd];
        FLAG(inc)=1;
    else
        commandArray_Idf= [CI(1,inc-1); CI(2,inc-1); CI(3,inc-1); CI(4,inc-1)];
        FLAG(inc)=0;
    end
else
    commandArray_Idf= [pitchCmd; rollCmd; yawCmd; vertCmd];
    FLAG(inc)=1;
end


move(p, 0.1, 'RotationSpeed', commandArray_Idf(3),'VerticalSpeed', commandArray_Idf(4),'roll', commandArray_Idf(2), 'pitch', commandArray_Idf(1));




% commandArray_Idf= [pitchCmd; rollCmd; yawCmd; vertCmd];


% TotalError
% Check if the total error is below the threshold
% If so, increment the ContGoal counter (indicates how long the goal is being maintained)
if TotalError<=0.1
    ContGoal=ContGoal+1
end


if GoalPt==1
if ContGoal==1500
    GoalPt=GoalPt+1;
    ContGoal=0;
end
end

if GoalPt>size(goalDesired,2)
    % Land the drone and save all relevant variables
    land(p)
    save('variables.mat','SS','Time','CI','DESIRED','v','kappa','gamma','c01','c02','zeta','delta')

end

Time1(inc)=toc;

end
