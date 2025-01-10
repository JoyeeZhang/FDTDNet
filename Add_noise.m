%% add noise with normalization (level)
data_num = 3000;
file_path = 'D:\KEYAN\MRE\code\shear_modulus_estimation\data\train\';
save_path = 'D:\KEYAN\MRE\code\shear_modulus_estimation\noise_0.03_data\train\';
noise_std = 0.03;
WaitBar = '【';
for n = 1:data_num
    load([file_path,num2str(n,'%06d'),'.mat'])
    n1 = size(wave,1);
    n2 = size(wave,2);
    ntimesteps = size(wave,3); 
    norm_wave = zeros(size(wave));
    % normalization
    for itimesteps = 1:ntimesteps
        max_i = max(max(wave(:,:,itimesteps)));
        min_i = min(min(wave(:,:,itimesteps)));
        norm_wave(:,:,itimesteps) = (wave(:,:,itimesteps) - min_i) ./ (max_i - min_i);
    end
    Gnoise = noise_std * randn(n1,n2,itimesteps);
    % add noise
    noise_wave = norm_wave + Gnoise;
    for itimesteps = 1:ntimesteps
        noise_wave(:,:,itimesteps) = noise_wave(:,:,itimesteps) .* mask;
    end
    wave = noise_wave;
    save([save_path,num2str(n,'%06d')],'wave','mu','mfre','driver','mask')
    % show progress
    if n < data_num
        if n <= 50
            w = [WaitBar '/' num2str((n / data_num) * 100, '%.2f') '%'];
        else
            w = [WaitBar num2str((n / data_num) * 100, '%.2f') '%'];
        end
    else
        w = [WaitBar '100%'];
    end
    clc
    ind = regexp(w,'\d','start');
    if n < data_num
        disp([w w(2 : ind(1) - 1)]);
    else
        disp([w w(2 : ind(1) - 1) '】']);
    end
    
    WaitBar = w(1 : ind(1) - 1);
end
%% add noise with normalization (snr)
data_num = 3000;
file_path = 'D:\zhangjiaying\shear_modulus_estimation\data\data0116\train\';
save_path = 'D:\zhangjiaying\shear_modulus_estimation\data\data0116_snr20_norm\train\';
snr = 20;
WaitBar = '【';
n1 = 256;
n2 = 256;
timesteps = 8;
for n = 1:data_num
    load([file_path,num2str(n,'%06d'),'.mat'])
    wave_noise = zeros(n1,n2,timesteps);
    max_wave = max(max(max(wave(:,:,:))));
    min_wave = min(min(min(wave(:,:,:))));
    norm_wave(:,:,:) = (wave(:,:,:) - min_wave)./(max_wave - min_wave);
    std_s = std(reshape(norm_wave(:,:,:),[256*256*8,1]));
    std_n = std_s ./ (10.^(snr/20));
    % std_n = 0;
    noise = std_n * randn(n1,n2,8);
    wave_noise(:,:,:) = norm_wave(:,:,:) + noise;
    wave = wave_noise;
    save([save_path,num2str(n,'%06d')],'wave','mu','mfre','index',"fov")
    % show progress
    if n < data_num
        if n <= 50
            w = [WaitBar '/' num2str((n / data_num) * 100, '%.2f') '%'];
        else
            w = [WaitBar num2str((n / data_num) * 100, '%.2f') '%'];
        end
    else
        w = [WaitBar '100%'];
    end
    clc
    ind = regexp(w,'\d','start');
    if n < data_num
        disp([w w(2 : ind(1) - 1)]);
    else
        disp([w w(2 : ind(1) - 1) '】']);
    end
    
    WaitBar = w(1 : ind(1) - 1);
end
%% add noise without normalization (level)
data_num = 1000;
file_path = 'D:\zhangjiaying\shear_modulus_estimation\data\data1117\test\';
save_path = 'D:\zhangjiaying\shear_modulus_estimation\test_noise\data1117_noise\test\';
noise_std = 0;
WaitBar = '【';
n1 = 256;
n2 = 256;
itimesteps = 8;
for n = 1:data_num
    load([file_path,num2str(n,'%06d'),'.mat'])
    Gnoise = noise_std * randn(n1,n2,itimesteps);
    % add noise
    noise_wave = wave + Gnoise;
    wave = noise_wave;
    save([save_path,num2str(n,'%06d')],'wave','mu','mfre','index',"fov")
    % show progress
    if n < data_num
        if n <= 50
            w = [WaitBar '/' num2str((n / data_num) * 100, '%.2f') '%'];
        else
            w = [WaitBar num2str((n / data_num) * 100, '%.2f') '%'];
        end
    else
        w = [WaitBar '100%'];
    end
    clc
    ind = regexp(w,'\d','start');
    if n < data_num
        disp([w w(2 : ind(1) - 1)]);
    else
        disp([w w(2 : ind(1) - 1) '】']);
    end
    
    WaitBar = w(1 : ind(1) - 1);
end
%% add noise without normalization (snr)
data_num = 10;
file_path = 'D:\zhangjiaying\shear_modulus_estimation\data\data1212\test\';
save_path = 'D:\zhangjiaying\shear_modulus_estimation\data\data1212_snr20\test_data\';
snr = 20;
WaitBar = '【';
n1 = 256;
n2 = 256;
timesteps = 8;
for n = 1:data_num
    load([file_path,num2str(n,'%06d'),'.mat'])
    wave_noise = zeros(n1,n2,timesteps);
    for i = 1:timesteps
        std_s = std(reshape(wave(:,:,i),[256*256,1]));
        std_n = std_s ./ (10.^(snr/20));
        noise = std_n * randn(n1,n2,1);
        wave_noise(:,:,i) = wave(:,:,i) + noise;
    end
    wave = wave_noise;
    save([save_path,num2str(n,'%06d')],'wave','mu','mfre','index',"fov")
    % show progress
    if n < data_num
        if n <= 50
            w = [WaitBar '/' num2str((n / data_num) * 100, '%.2f') '%'];
        else
            w = [WaitBar num2str((n / data_num) * 100, '%.2f') '%'];
        end
    else
        w = [WaitBar '100%'];
    end
    clc
    ind = regexp(w,'\d','start');
    if n < data_num
        disp([w w(2 : ind(1) - 1)]);
    else
        disp([w w(2 : ind(1) - 1) '】']);
    end
    
    WaitBar = w(1 : ind(1) - 1);
end
%% test different level noises
file_path = 'D:\zhangjiaying\shear_modulus_estimation\data\data1212\test\';
load([file_path, '000001.mat']);
[n1,n2] = size(mu);
timestep = 8;
snr = [5,10,15,20,25,30];
noise = zeros(6,n1,n2,timestep);
wave_noise = zeros(6,n1,n2,timestep);
for i = 1:timestep
    max_i = max(max(wave(:,:,i)));
    min_i = min(min(wave(:,:,i)));
    norm_wave(:,:,i) = (wave(:,:,i) - min_i) ./ (max_i - min_i);
    std_s = std(reshape(norm_wave(:,:,i),[256*256,1]));
    level = std_s ./ (10.^(snr./20));
    for n = 1:6
        noise(n,:,:,i) = level(n)*randn(n1,n1);
        wave_noise(n,:,:,i) = reshape(norm_wave(:,:,i),[256,256])+reshape(noise(n,:,:,i),[256,256]);
    end
end
sm = MRE_LFE(norm_wave,6,6,[0.2,0.2],1,mfre,"noncomp");
sm_noise = zeros(5,n1,n2);
for n = 1:6
    input = reshape(wave_noise(n,:,:,:),[256,256,8]);
    sm_noise(n,:,:) = MRE_LFE(input,6,6,[0.2,0.2],1,mfre,"noncomp");
end
c1 = [0 0 1]; 
c2 = [1 1 1]; 
c3 = [1 0 0]; 
n = 256; 
cmap = interp1([1 n/2 n], [c1; c2; c3], 1:n);
figure, 
subplot(3,7,1), imshow(norm_wave(:,:,3),[0,1]), colormap(cmap)
for n = 1:6
    wave_n = reshape(wave_noise(n,:,:,3),[256,256]);
    subplot(3,7,n+1), imshow(wave_n,[0,1]), colormap(cmap)
end
subplot(3,7,8), imshow(sm,[0 12]), colormap("parula")
for n = 1:6
    sm_n = reshape(sm_noise(n,:,:),[256,256]);
    subplot(3,7,n+8), imshow(sm_n,[0 12]), colormap(cmap)
end
subplot(3,7,15), imshow(mu/1000,[0 12]), colormap("parula")

%% test MRE data
c1 = [0 0 1]; 
c2 = [1 1 1]; 
c3 = [1 0 0]; 
n = 256; 
cmap = interp1([1 n/2 n], [c1; c2; c3], 1:n);
figure, 
for i = 1:8
    subplot(2,4,i), imshow(wave(:,:,i),[0 1])
end
colormap(cmap)
sm_LFE = MRE_LFE(wave,6,6,[0.2,0.2],1,mfre,"noncomp");
sm_DI = Direct_Inverse(wave,1000,[0.2,0.2],mfre,1000,4);
figure,
subplot(1,3,1), imshow(mu/1000,[0 12]), colormap('parula');
subplot(1,3,2), imshow(sm_LFE,[0 12]), colormap('parula');
subplot(1,3,3), imshow(sm_DI,[0 12]), colormap('parula')