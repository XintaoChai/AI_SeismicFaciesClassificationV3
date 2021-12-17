colormap(iColormapSeis);
axis tight;
xlim([1 max(xShotNsCoordinate)]);
ylim([1 max(yRecNrCoordinate)]);
zlim([min(zNzCoordinate) max(zNzCoordinate)]);
set(gca,'xTick',iInlineTick);
set(gca,'yTick',iCrossLineTick);
set(gca,'zTick',iTimeTick);
text(TimeXYZ(1), TimeXYZ(2), TimeXYZ(3),'Time (s)','rotation',90,'FontName','Arial','FontSize',iFontSize,'FontWeight','normal');
text(InlineXYZ(1), InlineXYZ(2), InlineXYZ(3),'Inline number','rotation',ilineRot,'FontName','Arial','FontSize',iFontSize,'FontWeight','normal');
text(CrosslineXYZ(1), CrosslineXYZ(2), CrosslineXYZ(3),'Crossline number','rotation',xlineRot,'FontName','Arial','FontSize',iFontSize,'FontWeight','normal');
shading interp;
set(gca,'xdir','reverse','ydir','reverse','zdir','reverse','tickdir','out','FontName',iFontName,'Fontsize',iFontSize,'Fontweight',iFontWeight);
set(gca,'box','on')
ax = gca;
ax.BoxStyle = 'full';
if opts.save
    print([opts.figures_path FigureName '.tif'],'-dtiff',opts.dpi);
end