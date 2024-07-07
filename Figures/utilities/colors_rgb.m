function [pu_box,high_PU,mid_PU,low_PU,color_screen,fb_green,...
    darkblue_muted,mix,perc,rew,gray_dots,light_gray,binned_dots,barface_green,...
    reg_color,dots_edges,dim_gray,fits_colors,gray_arrow,study2_blue] = colors_rgb()

    % function colors_rgb returns a set of predefined RGB color values used
    % for various graphical elements in plots and figures.
    %
    % OUTPUTS:
    %   pu_box:          RGB triplet for the color used in PU boxes
    %   high_PU:         RGB triplet for the color representing high PU levels (light green)
    %   mid_PU:          RGB triplet for the color representing mid PU levels (mid-level green)
    %   low_PU:          RGB triplet for the color representing low PU levels (dark green)
    %   color_screen:    RGB triplet for the gray color used for the trial screen
    %   fb_green:        RGB triplet for the green color used for feedback text
    %   darkblue_muted:  RGB triplet for a muted dark blue color
    %   mix:             RGB triplet for a mixed color used in plots
    %   perc:            RGB triplet for a specific color used in percentage plots
    %   rew:             RGB triplet for the color used in reward-related plots
    %   gray_dots:       RGB triplet for the light gray color used for single subject dots
    %   light_gray:      RGB triplet for a light gray color used in plots
    %   binned_dots:     RGB triplet for the bluish-green color used for binned analysis data
    %   barface_green:   RGB triplet for the green color used for bar faces in plots
    %   reg_color:       RGB triplet for a dark blue color used in regression plots
    %   dots_edges:      RGB triplet for the gray color used for dot edges
    %   dim_gray:        RGB triplet for a dark gray color used in plots
    %   fits_colors:     RGB triplet for a light blue color used in fit lines
    %   gray_arrow:      RGB triplet with transparency for the gray color used for arrows
    %   study2_blue:     RGB triplet for the blue color used in learning curves and bars in Study 2

    low_PU = [22, 77, 43]./255; % dark green 
    high_PU = [156, 196, 156]./255; % light green
    mid_PU = [71, 142, 104]./255; % mid level green
    
    light_gray = [184, 184, 184]./255; % light gray
    dim_gray = [140, 140, 140]./255; % dark gray
    
    reg_color = [92, 110, 129]./255; % dark blue color 
    
    rew = [18, 68, 138]./255; % 
    perc = [77, 124, 168]./255;
    mix = [127, 149, 179]./255;
    
    gray_dots = [220, 220, 220]./255; % gray color for single subject dots
    dots_edges = [184, 184, 184]./255; % gray for dots edges 
    
    binned_dots = [159, 210, 235]./255; % bluish green color for binned analysis data
    barface_green = [100, 119, 104]./255; % green for bars 
    
    study2_blue = [172, 207, 230]./255; % blue for learning curves, bars

    color_screen = [211,211,211]/255; % gray for trial screen
    pu_box = [178, 207, 204]./255; % green for pu boxes
    gray_arrow = [0.5 0.5 0.5 0.85]; % gray for arrows
    fb_green = [34,139,34]/255; % green for feedback text
    
    fits_colors = [117, 159, 199]/255; % light blue
    darkblue_muted = [37, 84, 156]/255; % dark blue    
end