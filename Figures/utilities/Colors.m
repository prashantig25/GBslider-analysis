classdef Colors
    % class Colors stores predefined RGB color values used for various
    % graphical elements in plots and figures. Access a color directly
    % as Colors.<name> (e.g. Colors.darkblue_muted) -- no need to
    % instantiate, and no need to pull every color out of a long
    % positional output list just to use one or two of them.

    properties (Constant)
        pu_box          = [178, 207, 204]./255; % green for pu boxes
        high_PU         = [156, 196, 156]./255; % light green, high PU levels
        mid_PU          = [71, 142, 104]./255;  % mid-level green, mid PU levels
        low_PU          = [22, 77, 43]./255;    % dark green, low PU levels
        color_screen    = [211, 211, 211]./255; % gray for the trial screen
        fb_green        = [34, 139, 34]./255;   % green for feedback text
        darkblue_muted  = [37, 84, 156]./255;   % muted dark blue
        mix             = [127, 149, 179]./255; % mixed color used in plots
        perc            = [77, 124, 168]./255;  % used in percentage plots
        rew             = [18, 68, 138]./255;   % used in reward-related plots
        gray_dots       = [220, 220, 220]./255; % light gray, single-subject dots
        light_gray      = [184, 184, 184]./255; % light gray
        binned_dots     = [159, 210, 235]./255; % bluish-green, binned analysis data
        barface_green   = [100, 119, 104]./255; % green for bar faces
        reg_color       = [92, 110, 129]./255;  % dark blue, regression plots
        dots_edges      = [184, 184, 184]./255; % gray for dot edges
        dim_gray        = [140, 140, 140]./255; % dark gray
        fits_colors     = [117, 159, 199]./255; % light blue, fit lines
        gray_arrow      = [0.5, 0.5, 0.5, 0.85]; % gray with transparency, arrows
        study2_blue     = [172, 207, 230]./255; % blue, Study 2 learning curves/bars
    end
end
