function add_3Dline(show_x,show_y,show_z,size_x,size_y,size_z)
    hold on
    show_x = unique([1, show_x, size_x]);
    show_y = unique([1, show_y, size_y]);
    show_z = unique([1, show_z, size_z]);
    list_x_z = linspace(1,size_z,length(show_x));
    for iy = 1:length(show_y)
        for i = 1:length(show_x)
            for j = 1:length(show_x)
                x_x(i,j) = show_x(j);
                x_y(i,j) = show_y(iy);
                x_z(i,j) = list_x_z(i);
            end
        end
        plot3(x_x,x_y,x_z,'Color','w','LineWidth',0.01 )%似乎并没有变细
    end
    
    list_z_y = linspace(1,size_y,length(show_x));
    for iz = 1:length(show_z)
        for i = 1:length(show_z)
            for j = 1:length(show_x)
                z_x(i,j) = show_x(j);
                z_z(i,j) = show_z(iz);
                z_y(i,j) = list_z_y(i);
            end
        end
        plot3(z_x,z_y,z_z,'Color','w')
    end
    
    list_z_x = linspace(1,size_x,length(show_y));
    for iz = 1:length(show_z)
        for i = 1:length(show_z)
            for j = 1:length(show_y)
                z_y(i,j) = show_y(j);
                z_z(i,j) = show_z(iz);
                z_x(i,j) = list_z_x(i);
            end
        end
        plot3(z_x,z_y,z_z,'Color','w')
    end
end