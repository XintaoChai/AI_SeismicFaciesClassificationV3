close all;clc;
opts.save = 1;
opts.dpi = '-r600';
iFontSize=8;
iFontWeight='normal';
iFontName='Arial';
iFigureWidth = 1.1666*3.33;
iFigureHight = 5.20;
iFigureUpDown =4.5;
iFigureLeftRight = 0.2311;
dFigureLeftRight = 8;
dFigureUpDown = 4.5;
inRow =10;
inColumn = 10;
a11 = [inColumn+2:inColumn*2];
iLayout = [ a11+0*inColumn a11+1*inColumn a11+2*inColumn a11+3*inColumn...
    a11+4*inColumn a11+5*inColumn a11+6*inColumn a11+7*inColumn a11+8*inColumn];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iWindowPosition1=[iFigureLeftRight + 0 * dFigureLeftRight iFigureUpDown - 0* dFigureUpDown iFigureWidth iFigureHight];
iWindowPosition2=[iFigureLeftRight + 0 * dFigureLeftRight iFigureUpDown - 1* dFigureUpDown iFigureWidth iFigureHight];
iWindowPosition3=[iFigureLeftRight + 1 * dFigureLeftRight iFigureUpDown - 0* dFigureUpDown iFigureWidth iFigureHight];
iWindowPosition4=[iFigureLeftRight + 1 * dFigureLeftRight iFigureUpDown - 1* dFigureUpDown iFigureWidth iFigureHight];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TimeXYZ =[1002.34151892835 43.8758160461766 1.38440299301907];
InlineXYZ = [763.112870518191 1584.83095426647 2.54437161724756];
CrosslineXYZ = [1146.30052512562 534.467373019263 2.506523987023];
MIOUXYZ =[932.197067499827 12.5491208097555 4.22199306303439];
xlineRot = -39.999;
ilineRot = 27;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xShotNsCoordinate=(1:1:IlinesPro);
yRecNrCoordinate=(1:1:XlinesPro);
zNzCoordinate=(0:ZsPro-1)*dt;
iInlineTick = unique(sort([200:100:500 xShotNsCoordinate(ilines_training) 700 max(xShotNsCoordinate)],'ascend'));
iCrossLineTick = unique(sort([1 150:150:600 yRecNrCoordinate(xlines_training) 900:150:1000 max(yRecNrCoordinate)],'ascend'));
iTimeTick = [0 0.2:0.2:max(zNzCoordinate)-0.2];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xShotNsSlices=[xShotNsCoordinate(1:5) xShotNsCoordinate(ilines_training-4:ilines_training)];
yRecNrSlices=[yRecNrCoordinate(1) yRecNrCoordinate(xlines_training-4:xlines_training)];
zNzSlices=[zNzCoordinate(end-4:end)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iColormapSeis = 'promax';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

FigureName = 'FigMergeAllXtest';
figure('name',FigureName);clf;
set(gcf,'color','white','windowstyle','normal','Units','inches','Position',iWindowPosition1);
set(gcf, 'PaperPositionMode', 'auto');
slice(xShotNsCoordinate,yRecNrCoordinate,zNzCoordinate,permute(Training_Test1_Test2_Image_Pro(:,:,:),[2 3 1]),xShotNsSlices,yRecNrSlices,zNzSlices,'cubic');
caxis(iCaxisX);
test3DCommonDown;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FigureName = ['FigMergeAllYpred' 'Best'];
figure('name',FigureName);clf;
set(gcf,'color','white','windowstyle','normal','Units','inches','Position',iWindowPosition2);
set(gcf, 'PaperPositionMode', 'auto');
slice(xShotNsCoordinate,yRecNrCoordinate,zNzCoordinate,permute(Training_Test1_Test2_Labels_Best(:,:,:),[2 3 1]),xShotNsSlices,yRecNrSlices,zNzSlices,'cubic');
caxis(iCaxisY);
text(MIOUXYZ(1), MIOUXYZ(2), MIOUXYZ(3),['mIoU = ' num2str(eval_scores_Best(:,4,eval_scores_ID),'%.4f')],'rotation',0,'FontName','Arial','FontSize',iFontSize,'FontWeight','normal');
test3DCommonDown_Y;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FigureName = 'FigMergeAllYtrueKnown';
figure('name',FigureName);clf;
set(gcf,'color','white','windowstyle','normal','Units','inches','Position',iWindowPosition3);
set(gcf, 'PaperPositionMode', 'auto');
slice(xShotNsCoordinate(1:size(TrainingData_Labels,3)),yRecNrCoordinate(1:size(TrainingData_Labels,2)),zNzCoordinate,permute(TrainingData_Labels(:,:,:),[2 3 1]),xShotNsSlices,yRecNrSlices,zNzSlices,'cubic');
caxis(iCaxisY);
test3DCommonDown_Y;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FigureName = ['FigMergeAllYpredKnownArea' 'Best'];
figure('name',FigureName);clf;
set(gcf,'color','white','windowstyle','normal','Units','inches','Position',iWindowPosition4);
set(gcf, 'PaperPositionMode', 'auto');
slice(xShotNsCoordinate(1:size(TrainingData_Labels,3)),yRecNrCoordinate(1:size(TrainingData_Labels,2)),zNzCoordinate,permute(Training_Test1_Test2_Labels_Best(:,1:size(TrainingData_Labels,2),1:size(TrainingData_Labels,3)),[2 3 1]),xShotNsSlices,yRecNrSlices,zNzSlices,'cubic');
caxis(iCaxisY);
text(MIOUXYZ(1), MIOUXYZ(2), MIOUXYZ(3),['mIoU = ' num2str(eval_scores_Best(:,4,eval_scores_ID),'%.4f')],'rotation',0,'FontName','Arial','FontSize',iFontSize,'FontWeight','normal');
test3DCommonDown_Y;

