clear all
close all
clc
ins_imagefiles = dir('*.png'); 
nfiles = length(imagefiles);
%15

%%
blue = [0;0;255];
green = [0;255;0];
red = [255;0;0];
i=1
%%
b=blue
g=green
r=red
%%
color_map = [1,0,0;0,1,0;0,0,1];
for i = 1: nfiles
    A = imread(strcat('added_prediction/',imagefiles(i).name));
    L = imread(strcat('pseudo_color_prediction/',imagefiles(i).name));
    B = labeloverlay(A,L,'Colormap',color_map);
    filename = strcat('rgb_overlaid_results/',imagefiles(i).name);
    imwrite(B,filename)
    disp(i)
end
%%
color_map = [1,0,0;0,1,0;0,0,1;0.1,0.5,0;0.1,0.5,0;0.3,0,0.5;0.8,0.5,0;0,0.2,0.5;0,0.7,0.5];
files = [6,11,18,69,77,83];
for i =1:length(files)
dad = imread(strcat('pseudo_color_prediction/',imagefiles(files(i)).name));
A = imread(strcat('rgb/',imagefiles(files(i)).name));
pan = imread(strcat('ins/',ins_imagefiles(files(i)).name));
out = uint8(zeros(1280,1920));
out(pan==1)=4;
out(pan==2)=5;
out(pan==3)=6;
out(pan==4)=7;
out(pan==5)=8;
out(pan==6)=9;
out_coord = find(out);
dad_coord = find(dad);
common = intersect(out_coord,dad_coord);

combo = uint8(out)+dad;
combo(common) = 0;
imtool(combo,[0,9])
B = labeloverlay(A,combo,'Colormap',color_map);
imtool(B)
filename = strcat('results/',imagefiles(i).name);
imwrite(B,filename)
end
%%
for i = 1: nfiles
out = zeros(1280,1920,3);
red = zeros(1280,1920);
green = zeros(1280,1920);
blue = zeros(1280,1920);
img = imread(imagefiles(i).name);
red(img==1)=255;
green(img==1)=0;
blue(img==1)=0;
red(img==2)=0;
green(img==2)=255;
blue(img==2)=0;
red(img==3)=0;
green(img==3)=0;
blue(img==3)=255;
out(:,:,1) = red;
out(:,:,2) = green;
out(:,:,3) = blue;
filename = strcat('results/',imagefiles(i).name);
imwrite(out,filename)
disp(i)
end
%%
A = imread('82.png');
img = imread('00082_rgb_FRONT.png');
out = zeros(1280,1920);
out(img(:,:,1)==255)=1;
out(img(:,:,2)==255)=2;
out(img(:,:,3)==255)=3;
out = uint8(out);
color_map = [1,0,0;0,1,0;0,0,1;0.1,0.5,0;0.1,0.5,0;0.3,0,0.5;0.8,0.5,0;0,0.2,0.5;0,0.7,0.5];
B = labeloverlay(A,out,'Colormap',color_map);
%imtool(B)
filename = strcat('results/','82.png');
imwrite(B,filename)

%%
imtool(out)

disp(imagefiles(i).name)
gr = rgb2gray(img);
bw = imbinarize(gr,0.098);
imtool(img)
reg = regionprops(bw);
cent = cat(1,reg.Centroid); %x,y format
siz = size(bw);  
CC = bwconncomp(bw);
L = labelmatrix(CC);
imtool(L,[min(min(L)),max(max(L))])
%%
poly=2
col=r
ff=CC.PixelIdxList;
[row,c] = ind2sub(size(bw),ff{1,poly});
for j=1:length(row)
  img(row(j),c(j),:) = col;
end
imtool(img)
%%
poly=3
col=g
ff=CC.PixelIdxList;
[row,c] = ind2sub(size(bw),ff{1,poly});
for j=1:length(row)
  img(row(j),c(j),:) = col;
end
imtool(img)
%%  
imwrite(img,imagefiles(i).name) 
i=i+1
disp(i)
imtool close all
%%
[sep,msg]=fopen('to_be_seperated.txt','at');
fprintf(sep,'%s\n',imagefiles(i).name)
i=i+1
imtool close all
%%
[seergb,msg]=fopen('check_rgb.txt','wt');
fprintf(seergb,'%s\n',imagefiles(i).name)
i=i+1
imtool close all
%%
img_new = img(:,:,3)==255;
CC = bwconncomp(img_new);
L = labelmatrix(CC);
imtool(L,[min(min(L)),max(max(L))])

ff=CC.PixelIdxList;
[r,c] = ind2sub(size(bw),ff{1,2});
for j=1:length(r)
  img(r(j),c(j),:) = g;
end
imtool(img)
