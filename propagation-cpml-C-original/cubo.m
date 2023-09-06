%Computational Geophysics Course
%Dr.Ing. Sergio Abreo.
%Loading the forward modeling
%November, 2019

%Set the dimensions
Nx=200;
Nz=200;
Nt=400;

% Adjust the path
fid = fopen('C:\Users\juan9\OneDrive - UNIVERSIDAD INDUSTRIAL DE SANTANDER\Academico\Maestria Geofisica\Tesis\Codigo\Juan-Diego\propagation-cpml-C-original\propagacion2.bin','rb');
prueba = fread(fid,'float32');
fclose(fid);
[prueba,pad_f1] = vec2mat(prueba,Nz);
prueba = prueba';

for k=1:Nt
    for j=1:Nx
        for i=1:Nz
            video(i,j,k)=prueba(i,j+(k-1)*Nx); 
        end
    end
end

figure(1)
for i=1:Nt
    imagesc(video(:,:,i)),colorbar % Loading the volume
    hold on
    xlabel('Distance (m)');
    ylabel('Depth (m)');
    str=sprintf('Iteracion %d',i);title(str);
    pause(0.001)
end