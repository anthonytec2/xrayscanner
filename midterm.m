close all
clearvars
clc
load('test_data.mat'); %Linear Attenuation Coeffcient Array
Nx = size(headct,1); %mm of imaging length in x
Ny = size(headct,2); %mm of imaging length in y
Nz = size(headct,3); %mm of imaging length in z
Mx = 200; %number of pixels x direction
My = 200; %number of pixels y direction
D = 2; %Physical size of a detector pixel in x or y mm
h = 50; %height above detector to bottom of imaging volume
H = h+Nz+200; %height above detector to x ray source
detector =zeros(Mx,My);
originDet=-1*([(Mx*D)/2,(My*D)/2 ]-[(Nx)/2,(Ny)/2 ]);
ep=[Nx/2, Ny/2, H];
settings.ep=ep;settings.Nx=Nx;settings.Ny=Ny;settings.originDet=originDet;settings.Nz=Nz;settings.Mx=Mx;settings.My=My;settings.D=D;settings.h=h;settings.H=H;
ubone = 0.573; %coeffcient corresponding to bone
ufat = 0.193; %coeffcient corresponding to fat
[x,y,z]=ind2sub(size(headct),find(headct > 0)); %keep already 0 vals 0
mu=zeros(Nx,Ny,Nz);
for i=1:length(x)
    mu(x(i),y(i),z(i)) = (((headct(x(i),y(i),z(i)) - 0.3)/(0.7))*(ubone-ufat))+ufat;
end
di=[];
ar=[];
for i=1:Mx
    for j=1:My
        pos=[originDet(1)+D*(i-1),originDet(2)+D*(j-1), 0];
        dir=(ep-pos)/norm(ep-pos);
        L=1;
        %disp(j+(i-1)*128);
        %settings.pos=pos;
        %settings.dir=dir;
        %plot_thing(settings)
        %drawnow
        %di(:,j+(i-1)*Mx)=dir;w=1;
        while pos(3)< h+Nz
            %ar{j+(i-1)*Mx}(:,w)=pos; w=w+1;
            [dist, id, pos] = onemove_in_cube_true(pos,dir);
            if pos(1) >=0 && pos(1) < Nx&&pos(2) >=0 && pos(2) < Ny && pos(3) >=h && pos(3) < h+Nz
                L=L*exp(-1*mu(1+floor(pos(1)),1+floor(pos(2)),1+floor(pos(3))-h)*dist);
            end
        end
        detector(i,j)=L;
    end
    %axesHandlesToChildObjects = findobj(gca, 'Type', 'line');
    %if ~isempty(axesHandlesToChildObjects)
    %    delete(axesHandlesToChildObjects);
    %end
end
imagesc((detector))

