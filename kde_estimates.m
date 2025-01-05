%% Task 6b
% Compute kernel density estimates and plot densities for c, phi, and theta

% Ensure that ML_estimates and QML_estimates variables are loaded or defined in the workspace
% ML_estimates and QML_estimates should be matrices with columns corresponding to [c, phi, theta]

% Parameter labels with Greek symbols for phi and theta
params = {'c', '\phi', '\theta'};  

% Define colors for ML and QML estimates
colors = {'b', 'r'};  % Blue for ML and Red for QML

% Define line styles for distinction
lineStyles = {'-', '--'};  % Solid for ML, Dashed for QML

% Create a figure with increased size for better visibility
figure('Color', 'w', 'Position', [100, 100, 1400, 600]);  % Width x Height adjusted for better layout

% Define tiled layout with custom padding
% Requires MATLAB R2019b or later
t = tiledlayout(1, 3, 'TileSpacing', 'Compact', 'Padding', 'Compact');

% Loop over parameters and create plots
for i = 1:3  % Loop over parameters (c=1, phi=2, theta=3)
    nexttile;  % Move to the next tile
    
    % --- Compute KDE for ML estimates
    [ML_density, ML_x] = ksdensity(ML_estimates(:, i));
    
    % --- Compute KDE for QML estimates
    [QML_density, QML_x] = ksdensity(QML_estimates(:, i));
    
    % --- Plot the ML KDE
    plot(ML_x, ML_density, 'LineWidth', 2, 'Color', colors{1}, 'LineStyle', lineStyles{1});
    hold on;  % Allow overlaying QML plot
    
    % --- Plot the QML KDE
    plot(QML_x, QML_density, 'LineWidth', 2, 'Color', colors{2}, 'LineStyle', lineStyles{2});
    
    % --- Format the plot
    title(['Kernel Density of ', params{i}], 'FontSize', 14, 'FontWeight', 'bold');
    xlabel(params{i}, 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Density', 'FontSize', 12, 'FontWeight', 'bold');
    
    % --- Add legend with transparent background
    legend({'ML', 'QML'}, 'Location', 'Best', 'FontSize', 12, 'Color', 'none');
    
    % --- Improve grid appearance
    grid on;
    grid minor;
    
    hold off;  % End overlay
end

% --- Add a super title to the figure with more space above the subplots
sg = sgtitle('Kernel Densities of ML and QML Estimates', 'FontSize', 16, 'FontWeight', 'bold');

% Adjust the position of the super title to have more space above
% Retrieve the current position of the super title
% Position vector: [left, bottom, width]
current_pos = sg.Position;

% Modify the Y-position to move the title higher
% Increase the Y-value to add more space above
sg.Position = [current_pos(1), current_pos(2) + 0.05, current_pos(3)];