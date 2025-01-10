% Create a folder for training data
% you can change the path for different data generation
file = 'D:\zhangjiaying\shear_modulus_estimation\data\train\';
total_data = 3000;
img_size = [256,256];
size1_range = [1,2,3,4,5];
size2_range = [1,2,4,8,16,32];
WaitBar = '【';
simulated_wave = zeros(img_size(1),img_size(2),1500);
for data_num = 1:total_data
    % generate simulated shear modulus map
    index1 = randi([1,5]);
    size1 = size1_range(index1);
    initial_mat1 = 0.5+9.5*rand(size1,size1);
    mat1 = imresize(initial_mat1, img_size);
    
    index2 = randi([index1,6]);
    size2 = size2_range(index2);
    initial_mat2 = 0.5+4.5*rand(size2,size2);
    mat2 = imresize(initial_mat2, img_size);
    mat3 = roundn(mat1 + mat2, -4);

    min_mat = min(min(mat3));
    max_mat = max(max(mat3));
    if max_mat == min_mat
        mat = mat3;
    else
        mat = 0.1 + 11.9*((mat3-min_mat)./(max_mat-min_mat));
    end
%     figure,
%     subplot(231), imshow(initial_mat1,[0 10]), colorbar;
%     subplot(232), imshow(mat1,[0 10]), colorbar;
%     subplot(233), imshow(mat3,[0 15]), colorbar;
%     subplot(234), imshow(initial_mat2,[0 5]), colorbar;
%     subplot(235), imshow(mat2,[0 5]), colorbar;
%     subplot(236), imshow(mat,[0 12]), colorbar;
%     colormap("parula")

    % generate displacement 
    driver = zeros(img_size);
    driver(1, :) = ones(1, img_size(2));
    weights = ones(img_size(1)+2, img_size(2)+2);
    abs_thick = min(floor(0.05*img_size(1)), floor(0.05*img_size(2)));
    abs_rate = 0.3 / abs_thick;
    for ix = 1:img_size(1)
        for iy = 1:img_size(2)
            i = 0;
            j = 0;
            if (iy < abs_thick + 1)
                i = abs_thick + 1 - iy;
            end
%             if (ix < abs_thick + 1)
%                 j = abs_thick + 1 - ix;
%             end
            if (img_size(2) - abs_thick < iy)
                i = iy - img_size(2) + abs_thick;
            end
            if (img_size(1) - abs_thick < ix)
                j = ix - img_size(1) + abs_thick;
            end
            if (i == 0 && j == 0)
                continue
            end
            rr = abs_rate * abs_rate * double(i*i + j*j);
            weights(1+ix, 1+iy) = exp(-rr);
        end
    end
    fov = [0.2, 0.2]; % field of view [m]
    rho = 1000 * ones(img_size); % density [kg/m3]
    mu = 1000 * mat; % shear modulus [N/m2]
    nx = img_size(1);
    ny = img_size(2);
    dx = fov(1)/nx;
    dy = fov(2)/ny;
    t_total = 0.2; % recording duration [s]
    nt = 2000;
    dt = t_total / nt;
    t = dt * linspace(1,nt,nt);
    mfre = 20 * randi([2,5]); % mechanical frequency [Hz]
    source_term = sin(2*pi*mfre*t);
    dt2rho_src = dt^2 / max(max(driver .* rho));
    force_x = 0*(source_term*dt2rho_src/(dx*dy)).' .* reshape(driver,[1 256 256]);
    force_y = 1*(source_term*dt2rho_src/(dx*dy)).' .* reshape(driver,[1 256 256]);
    
    % simulate wave field
    ux1 = zeros(nx+2, ny+2); % wave field at t-1
    uy1 = zeros(nx+2, ny+2);
    ux2 = zeros(nx+2, ny+2); % wave field at t
    uy2 = zeros(nx+2, ny+2);
    ux3 = zeros(nx+2, ny+2); % wave field at t+1
    uy3 = zeros(nx+2, ny+2);
    dxx = dx^2;
    dyy = dy^2;
    dxy = dx*dy;
    dyx = dx*dy;
    dtt = dt^2;
    
    % calculate
    offset = 0;
    wave = zeros(256,256,8);
    offsets = 8;
    figure,
    for it = 1:nt
        ux3 = zeros(size(ux3));
        uy3 = zeros(size(uy3));
        % derivatives
        % Uy
        duy_dyy = (uy2(2:end-1,1:end-2)-2*uy2(2:end-1,2:end-1)+uy2(2:end-1,3:end)) ./ dyy;
        duy_dxx = (uy2(1:end-2,2:end-1)-2*uy2(2:end-1,2:end-1)+uy2(3:end,2:end-1)) ./ dxx;
        duy_dyx = ((uy2(1:end-2,3:end)-uy2(3:end,3:end))-(uy2(1:end-2,1:end-2)-uy2(3:end,1:end-2))) ./ (4.0*dyx);
        % Ux
        dux_dxx = (ux2(1:end-2,2:end-1)-2*ux2(2:end-1,2:end-1)+ux2(3:end,2:end-1)) ./ dxx;
        dux_dyy = (ux2(2:end-1,1:end-2)-2*ux2(2:end-1,2:end-1)+ux2(2:end-1,3:end)) ./ dyy;
        dux_dxy = ((ux2(1:end-2,3:end)-ux2(3:end,3:end))-(ux2(1:end-2,1:end-2)-ux2(3:end,1:end-2))) ./ (4.0*dxy);
        % wave equations
        Py = mu.*(duy_dxx + duy_dyy);
        Px = mu.*(dux_dxx + dux_dyy);
%         Px = mu.*(dux_dyy + duy_dyx);
%         Py = mu.*(dux_dxy + duy_dxx);
        uy3(2:end-1,2:end-1) = Py./rho*dtt + 2*uy2(2:end-1,2:end-1) - uy1(2:end-1,2:end-1);
        ux3(2:end-1,2:end-1) = Px./rho*dtt + 2*ux2(2:end-1,2:end-1) - ux1(2:end-1,2:end-1);
        % add driver
        ux3(2:end-1,2:end-1) = ux3(2:end-1,2:end-1) + squeeze(force_x(it,:,:));
        uy3(2:end-1,2:end-1) = uy3(2:end-1,2:end-1) + squeeze(force_y(it,:,:));
        ux1 = ux2 .* weights;
        ux2 = ux3 .* weights;
        uy1 = uy2 .* weights;
        uy2 = uy3 .* weights;
        simulated_wave(:,:,it) = ux3(2:end-1,2:end-1);
%         % Show wave images
%         c1 = [0 0 1]; 
%         c2 = [0 0 0]; 
%         c3 = [1 0 0]; 
%         n = 256; 
%         cmap = interp1([1 n/2 n], [c1; c2; c3], 1:n);
%         if mod(it,10) == 0 
%             % u = sqrt(ux3.^2 + uy3.^2);
%             imagesc(uy3(2:end-1,2:end-1), [-1e-4 1e-4]);
%             colorbar; 
%             colormap(cmap); 
%             xlabel('m'); 
%             ylabel('m');
%             title(['Step = ',num2str(it),'/',num2str(nt),', Time: ',sprintf('%.4f',t(it)),' sec']);
%             drawnow;
%         end
        if it > nt - floor(nt/(t_total*mfre))
            if mod(it-nt+floor(nt/(t_total*mfre)),floor(1/mfre/offsets/dt)) == 0
                offset = offset+1;
                wave(:,:,offset) = uy3(2:end-1,2:end-1);
            end
        end
    end
%     max_wave = max(max(max(wave)));
%     min_wave = min(min(min(wave)));
%     wave = (wave-min_wave)./(max_wave-min_wave);
    index = data_num;
    save([file,num2str(data_num,'%06d')],'wave','mu','mfre','index','fov')

    %%%%% show generation progress %%%%%
    if data_num < total_data
        if data_num <= 50
            w = [WaitBar '/' num2str((data_num / total_data) * 100, '%.2f') '%'];
        else
            w = [WaitBar num2str((data_num / total_data) * 100, '%.2f') '%'];
        end
    else
        w = [WaitBar '100%'];
    end
    clc
    ind = regexp(w,'\d','start');
    if data_num < total_data
        disp([w w(2 : ind(1) - 1)]);
    else
        disp([w w(2 : ind(1) - 1) '】']);
    end
    
    WaitBar = w(1 : ind(1) - 1);
end
%% show different offsets wave images
c1 = [0 0 1]; 
c2 = [0 0 0]; 
c3 = [1 0 0]; 
n = 256; 
cmap = interp1([1 n/2 n], [c1; c2; c3], 1:n);
figure, 
for i = 1:8
    subplot(2,4,i), imshow(wave(:,:,i), [-9e-4 9e-4]), colormap(cmap)
end
norm_wave = zeros(256,256,8);
max_wave = max(max(max(wave)));
min_wave = min(min(min(wave)));
norm_wave = (wave-min_wave)./(max_wave-min_wave);
std_s = std(reshape(norm_wave(:,:,:),[256*256*8,1]));
snr = 20;
std_n = std_s ./ (10.^(snr/20));
noise = std_n * randn(256,256,8);
wave_noise(:,:,:) = norm_wave(:,:,:) + noise;
figure, 
for i = 1:8
    subplot(2,4,i), imshow(wave_noise(:,:,i), [0 1]), colormap(cmap)
end
% figure, imshow(mu,[]), colorbar, colormap("jet")
% sm_LFE = MRE_LFE(norm_wave,6,6,[fov,fov],1,mfre,"noncomp");
% sm_DI = Direct_Inverse(norm_wave,1000,fov,mfre,1000,4);
% figure, 
% subplot(131), imshow(sm_LFE,[0 20]), colormap("jet"), colorbar 
% subplot(132), imshow(sm_DI,[0 20]), colormap("jet"), colorbar
% subplot(133), imshow(mu/1000,[0 20]), colormap("jet"), colorbar
%% save gif
gif_file = 'D:\zhangjiaying\shear_modulus_estimation\simulated_wave.gif';
max_wave = max(max(max(simulated_wave)));
min_wave = min(min(min(simulated_wave)));
norm_simulated_wave = (simulated_wave-min_wave)./(max_wave-min_wave);
% figure, imshow(norm_simulated_wave(:,:,750),[])
for i = 1:1500
    if mod(i,10) == 0
        frame = norm_simulated_wave(:,:,i);
        [indexedFrame, cm] = gray2ind(frame, 256);
        rgbFrame = ind2rgb(indexedFrame, cmap);
        [imind, cm] = rgb2ind(rgbFrame, 256);
        if i == 10
            imwrite(imind, cm, gif_file, 'gif', 'LoopCount', inf, 'DelayTime', 0.1);
        else
            imwrite(imind, cm, gif_file, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
        end
    end
end
figure, imshow(mu/1000,[0,12]),colormap("parula")
