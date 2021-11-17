clear all

%first load linear track sessions
%M->M
%CAMKII
folders{1,1} = 'D:\vgat_hipp\miniscope\012';
folders{2,1} = 'D:\vgat_hipp\miniscope\013';
folders{3,1} = 'D:\vgat_hipp\miniscope\014';
folders{4,1} = 'D:\vgat_hipp\miniscope\022';
folders{5,1} = 'D:\vgat_hipp\miniscope\025';
folders{6,1} = 'D:\vgat_hipp\miniscope\026';
folders{7,1} = 'D:\vgat_hipp\miniscope\027';
folders{8,1} = 'D:\vgat_hipp\miniscope\028';

%vgat
folders{1,2} = 'D:\vgat_hipp\miniscope\010';
folders{2,2} = 'D:\vgat_hipp\miniscope\011';
folders{3,2} = 'D:\vgat_hipp\miniscope\016';
folders{4,2} = 'D:\vgat_hipp\miniscope\017';
folders{5,2} = 'D:\vgat_hipp\miniscope\018';
folders{6,2} = 'D:\vgat_hipp\miniscope\020';
folders{7,2} = 'D:\vgat_hipp\miniscope\023';
folders{8,2} = 'D:\vgat_hipp\miniscope\024';
folders{9,2} = 'D:\vgat_hipp\miniscope\029';
folders{10,2} = 'D:\vgat_hipp\miniscope\030';
folders{11,2} = 'D:\vgat_hipp\miniscope\031';
folders{12,2} = 'D:\vgat_hipp\miniscope\032';

usePCA = 0;
minVarAcc = 70;
smoothingAmount = 2;
firstPC = 2;

remLowSpeed = 0;
regressOutPosition = 1;

warning ('off','all');
%first cycle through behavior to see if minimal amount of exploration
for mouseNum = 1:size(folders,1)
    for assayNum = 1:size(folders,2)
        
        if isempty(folders{mouseNum,assayNum})
            distTraveled(mouseNum,assayNum) = nan;
            continue
        end
        
        cd(folders{mouseNum,assayNum})
        load('output_gbGetBehavior_2021_circPerm.mat','Tracking'); temp1 = Tracking.mouse_positionMS(:,1); clearvars Tracking
        
        %smooth x position
        temp1 = smoothdata(temp1,'gaussian',smoothingAmount);            
        
        distTraveled(mouseNum,assayNum) = nansum([abs(diff(temp1))]);
    end
end
stdDist = nanstd(reshape(distTraveled,1,[]));
meanDist = nanmean(reshape(distTraveled,1,[]));
distThresh = meanDist - stdDist; %less than mean - 1 SD remove
removeMice = find(distTraveled < distThresh);
for i=1:length(removeMice)
    folders{removeMice(i)} = [];
end

%then cycle through folders to see what the minimum number of 'good cells'
for mouseNum = 1:size(folders,1)
    for assayNum = 1:size(folders,2)
        
        if isempty(folders{mouseNum,assayNum})
            cellCount(mouseNum,assayNum) = nan;
            continue
        end
        
        cd(folders{mouseNum,assayNum})
        load('good_neurons.mat')
        cellCount(mouseNum,assayNum) = sum(good_neurons);
    end
end

minCellCount = nanmin(nanmin(cellCount));

for mouseNum = 1:size(folders,1)
    for assayNum = 1:size(folders,2)

            if isempty(folders{mouseNum,assayNum})
                MSE(mouseNum,assayNum) = nan;
                CorrAll(mouseNum,assayNum) = nan;                
                posPredTestAll{mouseNum,assayNum} = nan;
                continue
            end

            cd(folders{mouseNum,assayNum})
            load('output_gbGetBehavior_2021_circPerm.mat','neuron','Tracking');
            load('good_neurons.mat')
            
            mousePos = Tracking.mouseVelMS(:,1); %use mousePos bc already coded that way
            mousePos = smoothdata(mousePos,'gaussian',smoothingAmount);
            
            mouseTailbaseVel = Tracking.mouseTailbaseVelMS';
            mouseTailbaseVel = smoothdata(mouseTailbaseVel,'gaussian',smoothingAmount);

            sig_full = neuron.C_raw(find(good_neurons),:);    
            for i=1:size(sig_full,1)
               sig_full(i,:) = zscore(sig_full(i,:)); 
            end  
            
            %remove low speed samples
            if remLowSpeed == 1
                idxRem = find(mousePos < 2 | mouseTailbaseVel < 2);
                sig_full(:,idxRem) = [];
                mousePos(idxRem) = [];
            else
                idxRem = [];
            end
            
            if regressOutPosition == 1
            %regress out influence of position from neural activity PS NEW!
            xPos = Tracking.mouse_positionMS(:,1);
            xPos(idxRem) = [];
            for cellNum = 1:size(sig_full,1)
               
                  mdl = fitglm(xPos, sig_full(cellNum,:)');
                  sigPredict = predict(mdl,xPos);
                  sig_full(cellNum,:) = sig_full(cellNum,:) - sigPredict'; %subtract out the influence of speed.
                  
            end
            end
            
        for cellSelectIter = 1:100 %how many times a new batch of cells is selected for testing.
        for resampleNum = 1:100 
            
            %randomly choose the min cell count from neuron.C_raw
            tempIdx = randi(size(sig_full,1),minCellCount,1);
            sig = sig_full(tempIdx,:); clearvars tempIdx;
            
            if length(mousePos) < length(sig)
                sig = sig(:,1:end-1);
            end
            if length(mousePos) > length(sig)
                mousePos = mousePos(1:end-1);
            end
            
            if usePCA==1
               X = bsxfun(@minus,sig',mean(sig'));
               [coeff,score,latent,~,explained] = pca(X);
               temp = cumsum(explained); temp = min(find(temp > minVarAcc));
               sig = score(:,firstPC:temp)';
            end

        %split data 
        dataSegLength = floor(60 .* 7.5);  
        inbetweenSegs = floor(10 .* 7.5);
        iterTotal = floor(length(sig) ./ dataSegLength);
        %iterTotal = floor(iterTotal ./ 2);        

        trainIdx = zeros(1,length(sig)); 
        testIdx = zeros(1,length(sig));   
            for iterNum = 1:iterTotal       
               if bitget(iterNum,1) %odd
                    trainIdx(((iterNum-1).*dataSegLength)+1:(iterNum .* dataSegLength-inbetweenSegs)) = 1;
                    testIdx(((iterNum-1).*dataSegLength)+1:(iterNum .* dataSegLength-inbetweenSegs)) = 0;            
               else %even
                    trainIdx(((iterNum-1).*dataSegLength)+1:(iterNum .* dataSegLength-inbetweenSegs)) = 0;
                    testIdx(((iterNum-1).*dataSegLength)+1:(iterNum .* dataSegLength-inbetweenSegs)) = 1;            
               end 
            end

         sigTrain = sig(:,find(trainIdx));
         posTrain = mousePos(find(trainIdx));
         sigTest = sig(:,find(testIdx));
         posTest = mousePos(find(testIdx));

         mdl = fitglm(sigTrain',posTrain');
         posPredict = predict(mdl,sigTest');

         %posPredict(find(posPredict < 0)) = 0;
         %posPredict(find(posPredict > 600)) = 600;     

         error = posPredict-posTest;

         MSE_iter(cellSelectIter, resampleNum) = sqrt(nansum(error.^2) ./ length(sigTest)); %PS add
         Corr_iter(cellSelectIter, resampleNum) = corr(posPredict, posTest, 'rows','complete', 'type', 'Spearman');

        end 
        end
        
         MSE(mouseNum,assayNum) = nanmean(nanmean(MSE_iter,2)); %clearvars MSE_iter; %PS add
         CorrAll(mouseNum,assayNum) = nanmean(nanmean(Corr_iter,2)); %clearvars MSE_iter; %PS add       
         
         posPredTestAll{mouseNum,assayNum} = [posPredict,posTest]; %PS add
         
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%and do the same for a bootstrap distribution -- resample and
         %shuffle behavioral output each time%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         
         for resampleNum = 1:100
            
            %randomly choose the min cell count from neuron.C_raw
            tempIdx = randi(size(sig_full,1),minCellCount,1);
            sig = sig_full(tempIdx,:); clearvars tempIdx;
                        
            %mousePos = smoothdata(mousePos,'gaussian',2);

            if usePCA==1
               X = bsxfun(@minus,sig',mean(sig'));
               [coeff,score,latent,~,explained] = pca(X);
               temp = cumsum(explained); temp = min(find(temp > minVarAcc));
               sig = score(:,firstPC:temp)';
            end

            %rotate data here to make the bootstrap
            dataShiftAmount = floor(length(mousePos) ./ 100); %make one complete rotation of the data
            dataShiftAmount = dataShiftAmount .* resampleNum;
            mousePos_shift = [mousePos(dataShiftAmount:end);mousePos(1:dataShiftAmount-1)];
            
            %split data 
            dataSegLength = floor(60 .* 7.5);  
            inbetweenSegs = floor(10 .* 7.5);
            iterTotal = floor(length(sig) ./ dataSegLength);

            trainIdx = zeros(1,length(sig)); 
            testIdx = zeros(1,length(sig));   
            
            for iterNum = 1:iterTotal       
               if bitget(iterNum,1) %odd
                    trainIdx(((iterNum-1).*dataSegLength)+1:(iterNum .* dataSegLength-inbetweenSegs)) = 1;
                    testIdx(((iterNum-1).*dataSegLength)+1:(iterNum .* dataSegLength-inbetweenSegs)) = 0;            
               else %even
                    trainIdx(((iterNum-1).*dataSegLength)+1:(iterNum .* dataSegLength-inbetweenSegs)) = 0;
                    testIdx(((iterNum-1).*dataSegLength)+1:(iterNum .* dataSegLength-inbetweenSegs)) = 1;            
               end 
            end

             sigTrain = sig(:,find(trainIdx));
             posTrain = mousePos_shift(find(trainIdx));
             sigTest = sig(:,find(testIdx));
             posTest = mousePos_shift(find(testIdx));

             mdl = fitglm(sigTrain',posTrain');
             posPredict = predict(mdl,sigTest');

             posPredict(find(posPredict < 0)) = 0;
             %posPredict(find(posPredict > 600)) = 600;     

             error = posPredict-posTest;

             MSE_boot_iter(resampleNum) = sqrt(nansum(error.^2) ./ length(sigTest)); %PS add
             Corr_boot_iter(resampleNum) = corr(posPredict, posTest, 'rows','complete', 'type', 'Spearman');

       end
        
        MSE_boot{mouseNum,assayNum} = MSE_boot_iter; clearvars MSE_boot_iter;   
        Corr_boot{mouseNum,assayNum} = Corr_boot_iter; clearvars MSE_boot_iter;                     
    end
end


%% plot the MSE and correlation for prediction, camkii vs. vgat

[h,p_mse] = ttest2(MSE(:,1),MSE(:,2));
[h,p_corr] = ttest2(CorrAll(:,1),CorrAll(:,2));

%find number mice used in analysis per cell type
for mouseNum = 1:size(folders,1)
    for cellType = 1:size(folders,2)
        numMice(mouseNum,cellType) = ~isempty(folders{mouseNum,cellType});
    end
end
numMice = sum(numMice);

[pcorr(1), h] = signrank(CorrAll(:,1));
[pcorr(2), h] = signrank(CorrAll(:,2));

figure(135)
subplot(1,2,1)
bar(nanmean(MSE)); hold on;
errorbar(nanmean(MSE),(nanstd(MSE))./sqrt(numMice), 'LineStyle','none','Color','k')
box off;
title(['GLM mean error: n=', num2str(numMice(1)), '/', num2str(numMice(2))])
text(1.5,7.5,['p=' num2str(p_mse)],'Color','r');
ylabel('mean error: speed (cm/s)')
labels = {'CAMKII','vgat'};
set(gca,'XTickLabel',labels)

subplot(1,2,2)
bar(nanmean(CorrAll)); hold on;
errorbar(nanmean(CorrAll),(nanstd(CorrAll))./sqrt(numMice), 'LineStyle','none','Color','k')
box off;
title(['GLM mean r-value: n=', num2str(numMice(1)), '/', num2str(numMice(2))])
text(1.5,.7,['p=' num2str(p_corr)],'Color','r');
ylabel('mean r-value: speed')
labels = {'CAMKII','vgat'};
set(gca,'XTickLabel',labels)
ylim([0 1])
text(1,0,num2str(pcorr(1)),'Color','r')
text(2,0,num2str(pcorr(2)),'Color','r')

%% AND check to see if accuracy beats bootstrap distribution on a mouse-by-mouse basis
figure(2)
cntr = 1;
for mouseNum = 1:size(folders,1)
    for assayNum = 1:size(folders,2)
           if isempty(folders{mouseNum,assayNum})
               cntr = cntr+1;
               continue
           end
           
           subplot(size(folders,1),size(folders,2),cntr)
           hist(MSE_boot{mouseNum,assayNum}); box off; hold on;
           plot([MSE(mouseNum,assayNum),MSE(mouseNum,assayNum)],[0 30],'Color','r')
           ylim([0 30])
           significance = sum(MSE(mouseNum,assayNum) > MSE_boot{mouseNum,assayNum}) ./ 100;
           text(.5, 27, ['p=' num2str(significance)],'Color','r')
           cntr = cntr+1;
           xlabel('mean error')
           ylabel('permutation count')
           if assayNum==1
               title(['CAMKII ', num2str(mouseNum)])
           elseif assayNum==2
               title(['vgat ', num2str(mouseNum)])
           end
           xlim([0 10])
    end
end

figure(3)
cntr = 1;
for mouseNum = 1:size(folders,1)
    for assayNum = 1:size(folders,2)
           if isempty(folders{mouseNum,assayNum})
               cntr = cntr+1;
               continue
           end
           
           subplot(size(folders,1),size(folders,2),cntr)
           hist(Corr_boot{mouseNum,assayNum}); box off; hold on;
           plot([CorrAll(mouseNum,assayNum),CorrAll(mouseNum,assayNum)],[0 30],'Color','r')
           ylim([0 30])
           significance = sum(CorrAll(mouseNum,assayNum) < Corr_boot{mouseNum,assayNum}) ./ 100;
           text(.5, 27, ['p=' num2str(significance)],'Color','r')
           cntr = cntr+1;
           xlabel('Spearman r-value')
           ylabel('permutation count')
           if assayNum==1
               title(['CAMKII ', num2str(mouseNum)])
           elseif assayNum==2
               title(['vgat ', num2str(mouseNum)])
           end
           xlim([-1 1])
    end
end