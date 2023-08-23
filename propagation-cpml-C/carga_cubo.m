%Computational Geophysics Course
%Dr.Ing. Sergio Abreo.
%Loading the forward modeling
%November, 2019

%Set the dimensions
Nx=300;
Nz=100;
Nt=450;

% Adjust the path
%fid = fopen('/home/sergio/MEGA/Geofisica_computacional/Ejercicios_Ansi_C/2019-2/propagation/propagacion1.bin','rb');
fid = fopen('C:\Users\juan9\OneDrive - UNIVERSIDAD INDUSTRIAL DE SANTANDER\Academico\Maestria Geofisica\Tesis\Codigo\Juan-Diego\propagation-cpml-C/propagacion2.bin','rb');
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