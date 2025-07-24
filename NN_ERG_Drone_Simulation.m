close all
clear all
clc

% simulation script for a neural-network-based Explicit Reference Governor (ERG) applied to a drone navigating within non-convex constraint environments

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
% Derivative Gain Valuesla
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

% P matrix for the Lyapunov function
P = lyap(A_bar, 2*I);% 

% minimum and maximum eigenvalues of the P matrix
m_1=min(eig(P));
m_2=max(eig(P));

%feedforward gain
G=inv((-inv(A-B*K)*B)'*(-inv(A-B*K)*B))*(-inv(A-B*K)*B)'*[1 0 0;0 0 0;0 1 0;0 0 0;0 0 1;0 0 0];


% the initial states
X(:,1)=[0;0;-1.6;0;1.5;0];

% the initial guess for the applied reference
v0=[X(1,1);X(3,1)];% the same as the initial position 

%the goal point
r=[0;1.8];

height=1.5; % drone height in meters

%lipschitz (sensitivity of the equilibrium point to the change in applied
%reference)
mu=norm(-inv(A-B*K)*B*G);


% defining the wall constraints f1 & f2


x=-2:0.01:2;


b = 0; 
a = 0;
k1 = -4;
k2 = 4;
Beta = 1/((x(1))^2);



f1 =@(x) ((1/2*(a-k1))*Beta*(x-b).^2+(1/2)*(a+k1));

f2 =@(x) ((1/2*(a-k2))*Beta*(x-b).^2+(1/2)*(a+k2));



y1_values = f1(x);
y2_values = f2(x);





xi=0.5;% the distance from the constraint for which the repulsive force effect begins
delta=0.2; % the distance from the constraint from where the tightned constraint exists, and the repulsive force ...
% factor becomes equal to 1 

eta=0.01; % smoothing factor to avoid division by zero

lambda=0.01; % smoothing factor 2 in the adaptive kappa equation

dt = 0.01;    % Time step (seconds)

time = 0:dt:200; %simulation time
dt_v=0.1; % update the applied reference every dt_v sec




% Load the trained network
loaded_net = load('trained_network_drone.mat');
net = loaded_net.net;


tolerance =   0.531382759839911;

% the tree obstacles
c01=[0.5;-0.5]; % the center of the first circle
c02=[-0.5;0.5]; % the center of the second circle
radius=0.3;     % the radius for the circle



% these to plot the circles during the simulation






theta = linspace(0, 2*pi, 100); % Angle from 0 to 2*pi (360 degrees)
xcr1 = c01(1) + radius * cos(theta);
ycr1 = c01(2) + radius * sin(theta);

xcr_tight1 = c01(1) + (radius+delta) * cos(theta);
ycr_tight1 = c01(2) + (radius+delta) * sin(theta);

xcr_zeta1 = c01(1) + (radius+xi) * cos(theta);
ycr_zeta1 = c01(2) + (radius+xi) * sin(theta);


xcr2 = c02(1) + radius * cos(theta);
ycr2 = c02(2) + radius * sin(theta);

xcr_tight2 = c02(1) + (radius+delta) * cos(theta);
ycr_tight2 = c02(2) + (radius+delta) * sin(theta);

xcr_zeta2 = c02(1) + (radius+xi) * cos(theta);
ycr_zeta2 = c02(2) + (radius+xi) * sin(theta);




% Simulation Loop
for i = 1:length(time)

  




    if rem(time(i),dt_v)==0 % the applied reference update every dt_v sec

    if i==1

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
        gamma(i) = min([gamma1,gamma2, gamma3, gamma4, gamma5, gamma6, gamma7]);
        
       
        % Compute DSM
        Lyapunov=double((X(:,i)-[v0(1);0;v0(2);0;height;0])'*P*(X(:,i)-[v0(1);0;v0(2);0;height;0]));%add the height?
        DSM=max(gamma(i)-Lyapunov,0);
        
        %Compute NF rho_a is the attraction term and rho_r represent the
        %repulsive term
        rho_a=(r-v0)/max(norm(r-v0),eta);
       




        c1=v0(2)-f1(v0(1));  

        rho_r1= max(0, (xi - c1) / (xi - delta)) * [(-4*(a-k1)*(v0(1)-b))/max(norm(-4*(a-k1)*(v0(1)-b)),0.01); 1/norm(1)]  ;

        c2=f2(v0(1))-v0(2);  

        rho_r2= max(0, (xi - c2) / (xi - delta)) * [(Beta*(a-k2)*(v0(1)-b))/max(norm(Beta*(a-k2)*(v0(1)-b)),0.01);-1/norm(-1)];
    
        phi1=sqrt((v0(1)-c01(1))^2 + (v0(2)-c01(2))^2);

        c3=phi1 - radius;

        rho_r3=max( (xi - c3) /(xi - delta)  , 0) * [(v0(1)-c01(1))/phi1; (v0(2)-c01(2))/phi1];

        phi2=sqrt((v0(1)-c02(1))^2 + (v0(2)-c02(2))^2);

        c4=phi2 - radius;

        rho_r4=max( (xi - c4) /(xi - delta)  , 0) * [(v0(1)-c02(1))/phi2; (v0(2)-c02(2))/phi2];



        rho_r=rho_r1+rho_r2+rho_r3+rho_r4;

        Attraction_Field=rho_a+rho_r;
        g=DSM*Attraction_Field;



        % the equations for computing the adaptive kappa value
        distance_1 = min(sqrt((v0(1)-x).^2 + (v0(2) - (f1(x) + delta)).^2)); %minimum distance between v and the wall f1
        distance_2 = min(sqrt((v0(1)-x).^2 + (v0(2) - (f2(x) - delta)).^2)); %minimum distance between v and the wall f2

        vartheta(i)=abs(min(distance_1, min(distance_2,min(c3-delta,c4-delta)))); % the distance of the current equilibruim point from the virtual constraint

        kappa(i)=max(((sqrt(m_1)/(sqrt(m_1)+sqrt(m_2)))*vartheta(i)-(sqrt(m_2)/(sqrt(m_1)+sqrt(m_2)))*(norm(X(:,i)-[v0(1);0;v0(2);0;height;0])))/(mu*dt_v*max(norm(g),lambda)),0); %dynamic kappa
        vdot=kappa(i)*g;

        v(:,i)=v0+vdot*dt_v; % the applied reference update

        u(:,i)=-K*X(:,i)+G*[v(1,i);v(2,i);height]; % compute the control command
    else

         gamma1=net(v(:,i-1))-tolerance; %% compute the threshold for the wall constraints using neural network, % other options like dlnetwork with `predict` may offer faster execution



         % compute the threshold for the control input constraints
        gamma2 = ((K(1,:)*[v(1,i-1);0;v(2,i-1);0;height;0]-G(1,:)*[v(1,i-1);v(2,i-1);height]+0.06)^2)/(K(1,:)*inv(P)*K(1,:)');

        gamma3 = ((-K(1,:)*[v(1,i-1);0;v(2,i-1);0;height;0]+G(1,:)*[v(1,i-1);v(2,i-1);height]+0.06)^2)/(K(1,:)*inv(P)*K(1,:)');

        gamma4 = ((K(2,:)*[v(1,i-1);0;v(2,i-1);0;height;0]-G(2,:)*[v(1,i-1);v(2,i-1);height]+0.06)^2)/(K(2,:)*inv(P)*K(2,:)');

        gamma5 = ((-K(2,:)*[v(1,i-1);0;v(2,i-1);0;height;0]+G(2,:)*[v(1,i-1);v(2,i-1);height]+0.06)^2)/(K(2,:)*inv(P)*K(2,:)');



        % compute the threshold for the first obstacle constraint
        xv_telda1 = -radius*(([c01(1);0;c01(2);0;height;0] - [v(1,i-1);0;v(2,i-1);0;height;0])/norm([c01(1);0;c01(2);0;height;0] - [v(1,i-1);0;v(2,i-1);0;height;0]))+ [c01(1);0;c01(2);0;height;0];
        xv_bar = [v(1,i-1);0;v(2,i-1);0;height;0];

        partial_O1 = [2*(xv_telda1(1)-c01(1));  0; 2*(xv_telda1(3)-c01(2)); 0; 0; 0];

        gamma6 = ((partial_O1'*(xv_bar-xv_telda1))^2)/(partial_O1'*inv(P)*partial_O1);

        % compute the threshold for the second obstacle constraint
        xv_telda2 = -radius*(([c02(1);0;c02(2);0;height;0] - [v(1,i-1);0;v(2,i-1);0;height;0])/norm([c02(1);0;c02(2);0;height;0] - [v(1,i-1);0;v(2,i-1);0;height;0]))+ [c02(1);0;c02(2);0;height;0];
        xv_bar = [v(1,i-1);0;v(2,i-1);0;height;0];

        partial_O2 = [2*(xv_telda2(1)-c02(1));  0; 2*(xv_telda2(3)-c02(2)); 0; 0; 0];

        gamma7 = ((partial_O2'*(xv_bar-xv_telda2))^2)/(partial_O2'*inv(P)*partial_O2);


        % take the minimum among the thresholds

        gamma(i) = min([gamma1,gamma2, gamma3, gamma4, gamma5, gamma6, gamma7]);

        







       % compute the DSM
       Lyapunov=double((X(:,i)-[v(1,i-1);0;v(2,i-1);0;height;0])'*P*(X(:,i)-[v(1,i-1);0;v(2,i-1);0;height;0]));%add the height?

        DSM=max(gamma(i)-Lyapunov,0);


        %Compute NF rho_a is the attraction term and rho_r represent the
        %repulsive term
        rho_a=(r-v(:,i-1))/max(norm(r-v(:,i-1)),eta);
      

        c1=v(2,i-1)-f1(v(1,i-1));  

        rho_r1= max(0, (xi - c1) / (xi - delta)) * [(-4*(a-k1)*(v(1,i-1)-b))/max(norm(-4*(a-k1)*(v(1,i-1)-b)),0.01); 1/norm(1)]  ;

        c2=f2(v(1,i-1))-v(2,i-1);  

        rho_r2= max(0, (xi - c2) / (xi - delta)) * [(Beta*(a-k2)*(v(1,i-1)-b))/max(norm(Beta*(a-k2)*(v(1,i-1)-b)),0.01);-1/norm(-1)];

        phi1=sqrt((v(1,i-1)-c01(1))^2 + (v(2,i-1)-c01(2))^2);

        c3=phi1 - radius;

        rho_r3=max( (xi - c3) /(xi - delta)  , 0) * [(v(1,i-1)-c01(1))/phi1; (v(2,i-1)-c01(2))/phi1];


        phi2=sqrt((v(1,i-1)-c02(1))^2 + (v(2,i-1)-c02(2))^2);

        c4=phi2 - radius;

        rho_r4=max( (xi - c4) /(xi - delta)  , 0) * [(v(1,i-1)-c02(1))/phi2; (v(2,i-1)-c02(2))/phi2];



        rho_r=rho_r1+rho_r2+rho_r3+rho_r4;


        Attraction_Field=rho_a+rho_r;
        g=DSM*Attraction_Field;


        % the equations for computing the adaptive kappa value

        distance_1 = min(sqrt((v(1,i-1)-x).^2 + (v(2,i-1) - (f1(x) + delta)).^2)); %minimum distance between v and the wall f1
        distance_2 = min(sqrt((v(1,i-1)-x).^2 + (v(2,i-1) - (f2(x) - delta)).^2)); %minimum distance between v and the wall f2

        vartheta(i)=abs(min(distance_1, min(distance_2,min(c3-delta,c4-delta)))); % the distance of the current equilibruim point from the virtual constraint



        kappa(i)=max(((sqrt(m_1)/(sqrt(m_1)+sqrt(m_2)))*vartheta(i)-(sqrt(m_2)/(sqrt(m_1)+sqrt(m_2)))*(norm(X(:,i)-[v(1,i-1);0;v(2,i-1);0;height;0])))/(mu*dt_v*max(norm(g),lambda)),0); %dynamic kappa
        

        vdot=kappa(i)*g;

        v(:,i)=v(:,i-1)+vdot*dt_v; % the applied reference update

        u(:,i)=-K*X(:,i)+G*[v(1,i);v(2,i);height]; % compute the control command
    end
    
    else
        v(:,i)=v(:,i-1);
        kappa(i)=kappa(i-1);
        u(:,i)=-K*X(:,i)+G*[v(1,i);v(2,i);height];

    end
    % Update State Using State Space Model
    x_dot(:,i) = A * X(:,i) + B * u(:,i);
    X(:,i+1) = X(:,i) + x_dot(:,i) * dt;


     cla; % clear plot
    plot(x,  f1(x),'r', 'LineWidth', 2); % f1 in red
    plot(x, f1(x)+delta,  'y', 'LineWidth', 0.5); % f1 in red
    % plot(x, f1(x)+xi,  '--k', 'LineWidth', 0.5); % f1 in red

    hold on;
    plot(x,  f2(x),'b', 'LineWidth', 2); % f2 in blue
    plot(x, f2(x)-delta,  'm', 'LineWidth', 0.5); % f2 in blue
    % plot(x, f2(x)-xi,  '--k', 'LineWidth', 0.5); % f1 in red


     plot(X(1,i),X(3,i), 'or', 'MarkerFaceColor', 'g'); %  drone as a green circle
     % plot(v(1,:),v(2,:))
     plot(v(1,i), v(2,i), 'or', 'MarkerFaceColor', 'r');  %  v point as a red circle

     plot(r(1),r(2),'*', 'LineWidth', 1)


     % Plot the first circle
    plot(xcr1, ycr1, 'b');
    % plot(xcr_tight1, ycr_tight1, 'c');
    % plot(xcr_zeta1,ycr_zeta1, '--k');



    % Plot the second circle
    plot(xcr2,ycr2, 'b');
    % plot(ycr_tight2,xcr_tight2, 'c');
    % plot(xcr_zeta2,ycr_zeta2, '--k');
    
    % grid minor
    
    xlabel('X axis')
    ylabel('Y axis')
    drawnow;  % Force plot update
end
